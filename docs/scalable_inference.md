# Scalable Inference — how we made MUSE-CN/FINDER run on million-node graphs

Date: 2026-06-21. This documents the surgical, **additive** changes that let the
already-trained models do inference on graphs up to Flickr (1,624,991 nodes /
15,473,043 edges) within a 64 GB-RAM / 24 GB-4090 box, **without retraining and
without changing any published result** (the chunked path is numerically
identical to the original — verified bit-for-bit, see §5).

Companion docs: `implementation_context.md` (overall pipeline + model labels),
`run_inference.py` (the runner).

---

## 0. Executive summary

**Problem.** The trained models OOM on large graphs — not because the model is big
(1.8M params), but because three tensors in the forward pass grow with the *graph*:
the decoder outer-product `[n, d, d]` (~419 GB at n=1.6M), the cross-attention
`[5, n, d]`, and the GraphSAGE message tensor `[E, d]` (~15 GB). The last one
OOM-killed the first Flickr run at 59 GB.

**Fix (exact, no quality loss).** All three are independent per node, so:
- the **decoder** and **cross-attention** are evaluated in **node chunks** (e.g.
  8000 nodes at a time) — identical arithmetic, tiny peak;
- the **GraphSAGE aggregation** is rewritten as a **sparse-dense matmul**
  (`torch.sparse.mm(A, x)`) that computes the same neighbor-sum without ever
  building the per-edge `[E, d]` tensor.

Result: peak memory 59 GB OOM → ~32 GB; Flickr runs. Bit-identical to the original
on test graphs (ANC = manuscript values).

**Why the "virtual full-graph node" does not break this.** FINDER/MUSE-CN add a
virtual node connected to all nodes that carries a *global summary* of the graph;
each node's representation concatenates its own local embedding with this global
embedding. That sounds like "every node needs all nodes" — but the global summary
is a **single shared `d`-vector**, produced by a cheap **reduction** (`sum` over all
node embeddings, `custom_global_add_pool` → `[1, d]`) and then **broadcast** to
every node (`rep_global` → `rep_y = [n, d]`, all rows identical). It is computed
**once per forward, before chunking**. So each node decodes from *(its own
embedding + the same precomputed global vector)* and remains per-node independent —
every chunk just reuses that one global vector. The "needs all nodes" part collapses
to one vector early (O(n)); the expensive per-node work happens after and chunks
cleanly. That is why the chunked path stays exact.

---

## 1. The problem

The trained models run fine on the benchmark graphs (≤ ~34k nodes) but OOM on the
NewMUSE-CNdata graphs. The cause is **not** model size (1.8M params). It is three
*per-node / per-edge intermediates* materialized in the forward pass that scale
with graph size:

| # | Intermediate | Where (original code) | Size at n=1.6M, d=256, E=15.5M |
|---|---|---|---|
| 1 | Decoder outer-product `[n, d, d]` | `FINDER_DQN.test_forward` (models.py) | **~419 GB** |
| 2 | Cross-attention Q/K/V `[5, n, d]` | `MEGA_Encoder.forward` → `CrossAttentionModule` (encoders.py) | ~8 GB each, several transient |
| 3 | GraphSAGE message tensor `[E, d]` | `CustomSAGEConv.forward` → PyG `propagate` (layers.py) | **~15 GB** |

Wall #3 is what OOM-killed the first Flickr attempt (kernel `oom-kill`,
anon-rss 58.9 GB). Walls #1 and #2 exist in FINDER and MUSE-CN alike (#1 is the
S2V/FINDER decoder; #2 is MUSE-CN-only).

Key insight: **all three are independent per node** (each node's Q-value uses only
its own embedding + the shared global/virtual-node vector; cross-attention is over
the 5 encoder tokens *within* a node; SAGE sum-aggregation is a linear operator).
So they can be computed in **node chunks** (#1, #2) or as a **sparse matmul** (#3)
with the *exact same arithmetic* — no approximation, no quality loss.

Quantization was considered and rejected as the primary lever: 419 GB → fp16 =
210 GB, still impossible, and it would perturb the headline model's numbers.
Chunking is mandatory *and* sufficient.

---

## 2. Files changed and exactly what

Four pre-existing repo files received **new methods only** (existing methods
untouched; new code is guarded/opt-in so default behavior is byte-identical). One
new file (the runner) orchestrates.

### 2.1 `py_modules/layers.py` — fix wall #3 (SAGE message tensor)
- **Added** `CustomSAGEConv._forward_spmm(self, x, edge_index)`: computes the
  identical sum-aggregation `out[i] = Σ_{j→i} x[j]` as a sparse-dense matmul
  `torch.sparse.mm(A, x)` where `A[target, source] = 1` (indices
  `[edge_index[1], edge_index[0]]`), then the same `self.lin` / `normalize`.
  No `[E, d]` message tensor is ever created — peak drops from +15 GB to ~+2 GB.
- **Added** a one-line guard at the top of the existing `CustomSAGEConv.forward`:
  `if getattr(self, 'use_spmm_aggr', False): return self._forward_spmm(...)`.
  Default (`use_spmm_aggr` unset/False) → original PyG `propagate` path, unchanged.
- Equivalence: with PyG's default flow, `propagate(aggr='sum')` message `x_j` is the
  source feature aggregated at the target node = `A @ x`. Identical regardless of
  edge directionality, because `A` is built from whatever columns `edge_index`
  contains.

### 2.2 `py_modules/models.py` — fix wall #1 (decoder outer-product)
- **Added** `FINDER_DQN.test_forward_chunked(self, data, return_embedding=False,
  dtype=torch.float, node_chunk_size=20000)`: same math as `test_forward`, but the
  decoder (the `[n, d, d]` outer product + contraction with `cross_product`, then
  `h1_weight`/`last_w`/`rep_aux`) is evaluated in node chunks of `node_chunk_size`,
  so only `[chunk, d, d]` exists at a time. Calls the encoder's `forward_chunked`
  if present (MEGA), else the normal encoder forward (single-encoder FINDER).
- Existing `test_forward`, `train_forward`, `_forward`, `forward` untouched.

### 2.3 `py_modules/encoders.py` — fix wall #2 (cross-attention)
- **Added** `MEGA_Encoder.forward_chunked(self, data, node_chunk_size=20000,
  dtype=torch.float)`: runs the 5 encoders full-graph (each emits `[n, d]`), then
  applies the distillation module (`cross_attention` or `mlp`) in node chunks so the
  `[5, n, d]` stack / attention Q,K,V are never materialized whole. Output equals
  `forward()` (per-node independent; final `F.normalize` is per-row).
- Existing `MEGA_Encoder.forward` and the GraphSAGE encoder untouched.

### 2.4 `Q_CNDP.py` — driver
- **Added** `Q_CNDP_Agent.predict_chunked(self, g_list, covered,
  node_chunk_size=20000, return_embedding=False)`: mirrors `predict` (batch=1) but
  calls `DQN.test_forward_chunked`; moves results to CPU/numpy with the same
  idx_map remapping; frees per-graph tensors + `empty_cache` between graphs.
- **Added** `Q_CNDP_Agent.calc_solution_nx_chunked(self, g, budget, step_size=1,
  node_chunk_size=20000, node_attributes=None, print_progress=False)`: same
  autoregressive node-removal rollout as `calc_solution_nx`, but (a) sets
  `use_spmm_aggr = True` on **every** `CustomSAGEConv` in `self.DQN` (enabling §2.1),
  (b) scores via `predict_chunked`, (c) **re-raises CUDA OOM** instead of swallowing
  it (other errors still return partial progress) so the runner can fall back to CPU.
- Existing `predict`, `predict_with_current_qnet`, `calc_solution_nx` untouched.

### 2.5 `2026_06_21_Dissertation_context_data/run_inference.py` — orchestration (new file)
- `CHUNK_NODE_THRESHOLD = 100000`: auto-enable the chunked path above this; flags
  `--chunked` / `--no-chunked` override. Below it, runs are byte-identical to before.
- `--node-chunk N` (default 20000; we use **8000** on GPU for headroom).
- `--step-size K`: removals per rollout step. Auto = 1% of budget above
  `BIG_GRAPH_NODE_THRESHOLD = 20000` (fewer, larger steps → tractable wall-time on
  giants), else 1.
- **GPU-first with CPU fallback**: chunked path tries CUDA; on CUDA OOM
  (`run_one_model` catch) it rebuilds the agent on CPU and retries. Works because
  the spmm fix removed the 15 GB tensor, so most graphs now fit the 4090.
- Batch + crash-safety helpers (added later same day): lazy graph loading
  (`load_graph` reads `spec['path']`), `--all-new-graphs` (every graph
  largest→smallest, one in memory at a time), per-(method,graph) **atomic**
  checkpoint to `results.csv` (temp file + `os.replace`), and **resume** (skip pairs
  already in `results.csv`).

---

## 3. Data flow (chunked path)

```
calc_solution_nx_chunked(g, budget, step_size, node_chunk_size)
  ├─ set use_spmm_aggr=True on all CustomSAGEConv          (§2.1 enables sparse agg)
  └─ rollout loop (until budget):
       predict_chunked(g_list, covered, node_chunk_size)
         └─ DQN.test_forward_chunked(data, node_chunk_size)   (§2.2)
              ├─ encoder.forward_chunked(data, node_chunk_size) (§2.3, MEGA)
              │     └─ 5× GraphSAGE encoder forward
              │           └─ CustomSAGEConv.forward → _forward_spmm (§2.1)
              └─ decoder evaluated in node chunks               (§2.2)
       remove top step_size nodes, update state, repeat
```

Peak working set after the fix: ~2 GB (spmm out) + 5×`[n,d]` (~8.5 GB at 1.6M) +
one `[chunk,d,d]` (~5 GB at chunk 20000, ~2 GB at 8000) ≈ **15–20 GB**, vs the
419 GB / 15 GB walls before.

---

## 4. How to use

```bash
# auto (chunked kicks in >100k nodes), GPU-first + CPU fallback
python 2026_06_21_Dissertation_context_data/run_inference.py \
    --new-graph Flickr \
    --models MTSSL_MEGA_CrossAttention_1l_freeze_BA_FINDER_noProc \
    --chunked --node-chunk 8000 --export flickr_muse_cn

# full sweep, all graphs largest->smallest, both focus models, crash-safe + resumable
python 2026_06_21_Dissertation_context_data/run_inference.py \
    --all-new-graphs --suite focus --chunked --node-chunk 8000 --export 2026_06_21
```
Always launch long jobs with `python -u` (else `nohup` block-buffers stdout and
hides the `step_size=` / `enabled spmm` confirmation lines).

---

## 5. Verification (no quality loss)

Compared `calc_solution_nx` (original) vs `calc_solution_nx_chunked` (spmm + chunked
decoder + chunked attention), same checkpoints:

| Graph | nodes | result |
|---|---|---|
| Bovine | 121 | identical solution, ANC 0.143193 == 0.143193 (= manuscript 14.32) |
| USAir97 | 332 | ANC 0.4642 (= manuscript 46.42) |
| openflights | 1491 | identical, ANC 0.400755 == 0.400755 (= manuscript 40.08) |

With a single node chunk the solution is **bit-identical**. With many tiny chunks,
greedy argmax can break a near-tie differently (float reordering, same effect as
CPU-vs-GPU) — measured ANC delta ≤ 5e-5, i.e. equal quality. Use a reasonably large
`--node-chunk` (≥ ~8000) and it is effectively identical.

Flickr / MUSE-CN confirmed running end-to-end: graph loaded, spmm enabled,
`step_size=1624`, peak RAM ~32 GB (vs the 59 GB OOM-kill before the fix),
~1624 nodes per ~3.3 min step.

---

## 6. Scalable classical baselines (HDA, CI)

The classical heuristics did not scale either: `matrix_collective_influence`
(`py_modules/baselines.py`) builds a **dense n×n adjacency** (`todense()`) plus
`A**radius`, so it OOMs above ~60–70k nodes; `agile_HDA` chains `G.subgraph` and
updates one node per recompute (slow on the giants). `degree` / `random` already
scale.

**New file `2026_06_21_Dissertation_context_data/scalable_baselines.py`** adds
*additive, batched, residual* drop-ins, used by `run_inference.py` for the `HDA`
and `CI` baselines (`degree`/`random` unchanged):

- `scalable_HDA(G, budget, step_size)` — each round take the `step_size`
  highest-degree alive nodes, remove them, **decrement neighbors' degrees**
  (residual update), repeat. No graph copy (just a degree dict + alive set).
  `step_size == 1` reproduces exact greedy `agile_HDA` (verified equal set).
- `scalable_CI(G, budget, step_size, radius=2, node_chunk_size=...)` — each round
  build the **sparse CSR** adjacency of the *residual* graph and score every node
  with CI computed in **row chunks**: `boundary = (A**radius) > 0`,
  `score_i = (k_i-1)·Σ_{j∈boundary}(k_j-1)` — the **exact same maths** as
  `matrix_collective_influence`, but `A[i:e] @ A` is done per row block so neither
  `.todense()` nor a full `A**radius` is ever formed. Remove the top `step_size`,
  recompute on the residual graph, repeat. On the unweighted CNDP graphs the
  per-round scores are **bit-identical** to `matrix_collective_influence`
  (verified, max abs diff 0.0 on Bovine); selection among equal-score ties may
  differ (same float-reorder caveat as the models).

Both run in `budget / step_size` recomputes instead of `budget` (the same "fewer,
larger steps" lever the learning models use on big graphs), and `step_size` is the
same per-graph value (`1%` of budget above `BIG_GRAPH_NODE_THRESHOLD`, else 1).
`run_one_baseline` routes `HDA`/`CI` here and threads `--node-chunk` (CI row block)
and `--step-size`.

```bash
# baselines for every graph, largest->smallest, crash-safe + resumable
python -u 2026_06_21_Dissertation_context_data/run_inference.py \
    --all-new-graphs --models --baselines degree HDA CI \
    --node-chunk 8000 --export 2026_06_21
```
(`--models` with no value runs baselines only; add a `--suite` / `--models …` to do
both in one resumable sweep.)

**Caveat — CI on the two million-node hub graphs (Flickr, Youtube).** CI radius-2
cost is `Σ deg²` per recompute; on heavy-tailed hub graphs that is large regardless
of implementation. The sparse row-chunked version no longer *crashes* (bounded
memory) and the mid-size graphs that used to OOM (Epinions 75k, Facebook 63k,
Gnutella 62k) now run fine, but Flickr/Youtube CI may still be slow — reduce
`--node-chunk` if a hub block spikes memory, or run those two last.
