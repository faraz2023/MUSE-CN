# MUSE-CN — Implementation Context & Inference Guide

This document explains how the MUSE-CN codebase runs inference, how to use the
additive runner added for the dissertation extension, how to experiment on new
graphs, and what every baseline / model checkpoint corresponds to in the
manuscript (`muse_cn.tex`).

Everything here is **additive**. No existing code, checkpoint, or result was
modified. The new code lives only in this folder:
`2026_06_21_Dissertation_context_data/run_inference.py`, and writes only under
`2026_06_21_Dissertation_context_data/Exports/`.

---

## 1. What the manuscript is

MUSE-CN is an RL (deep Q-learning) agent for the **Critical Node Detection
Problem (CNDP)**: pick a budget `b` of nodes whose removal most degrades graph
connectivity (measured by NPC — normalized pairwise connectivity). The agent
scores nodes, removes the highest-scored one, re-scores, and repeats
(autoregressive rollout) until the budget is spent.

Architecture: 5 self-supervised pretrained GraphSAGE encoders → a cross-attention
distillation module → an MLP decoder that outputs a Q-value (criticality score)
per node. Two metrics are reported:

- **ANC (%)** — accumulated normalized connectivity over the removal rollout
  (lower = faster disconnection = better). *Stored in `results.csv` as a fraction;
  multiply by 100 to match manuscript tables.*
- **NPC** — final connectivity after the full budget (lower = better). In
  `results.csv` this is `end_pairwise_connectivity`.

---

## 2. Code flow (inference path)

Entry → `2026_06_21_Dissertation_context_data/run_inference.py` (mirrors the
original `eval_test.py` but parameterized and writing to its own export dir).

```
run_inference.py
  ├─ chdir(REPO_ROOT)                  # params.json paths are relative to repo root
  ├─ build_graph_specs()              # --test / --benchmark / --new-graph / --edgelist
  │    └─ prep_G_for_tensor(path)     # py_modules/morPOP_utils.py
  │         read edgelist → drop deg-0 nodes → relabel 0..n-1
  └─ evaluate(specs, models, baselines, export_path)
       for each graph:
         write G.el, compute start NPC
         for each model exp_name:
           find_best_model_path(exp)          # lowest validation connectivity ckpt
           params = load experiments/MEGA_integration/<exp>/params.json
           agent  = Q_CNDP_Agent(params)      # Q_CNDP.py — builds encoder+decoder
           agent.LoadModel(best_ckpt)         # loads trained weights
           removed = agent.calc_solution_nx(G, budget, step_size)
                       # Q_CNDP.py:1160 — greedy autoregressive node removal
           ANC = calc_ANC_score(...)          # py_modules/utils.py
           write <exp>_sol.txt, append row
         for each baseline: get_baseline_sol(...)  # py_modules/baselines.py
         write results.csv
```

Key source files (all pre-existing, unchanged):

| File | Role |
|---|---|
| `Q_CNDP.py` | `Q_CNDP_Agent`: builds model from params, `LoadModel`, `calc_solution_nx` rollout |
| `py_modules/morPOP_utils.py` | `prep_G_for_tensor` — edgelist → clean nx.Graph |
| `py_modules/utils.py` | `calc_ANC_score`, `calc_pairwise_connectivity` |
| `py_modules/baselines.py` | `get_baseline_sol` — classical heuristics |
| `py_modules/encoders.py`, `models.py`, `layers.py` | GraphSAGE encoders, MEGA multi-encoder, cross-attention distillation |
| `experiments/MEGA_integration/<exp>/params.json` | full config for each trained model |
| `experiments/MEGA_integration/<exp>/levels/.../models/*.ckpt` | trained weights |
| `pretraining_module/SSL_models/SSL_v1_noProcAttr_d128/...` | frozen SSL encoder weights loaded by MEGA |

A checkpoint is chosen by `find_best_model_path`: it scans every level's
`level_log.json` and picks the iteration with the lowest validation connectivity
(`vc`).

---

## 3. How to use the runner

Always inside the conda env: `conda activate learnCNDP`. Run from anywhere; the
script `chdir`s to the repo root itself.

```bash
# Smoke test — all 7 model options on the small Bovine benchmark.
#   -> Exports/test/
python 2026_06_21_Dissertation_context_data/run_inference.py --test

# All models on an existing benchmark graph (Data/Data_benchmark/<name>.txt).
python 2026_06_21_Dissertation_context_data/run_inference.py \
    --benchmark USAir97 --export usair_run

# A NEW larger instance from NewMUSE-CNdata/ (the dissertation extension goal).
python 2026_06_21_Dissertation_context_data/run_inference.py \
    --new-graph Enron --export enron_run

# Any arbitrary edgelist.
python 2026_06_21_Dissertation_context_data/run_inference.py \
    --edgelist /path/graph.txt --label mygraph --budget-frac 0.1

# Restrict to specific models and add classical baselines.
python 2026_06_21_Dissertation_context_data/run_inference.py --benchmark USAir97 \
    --models ORIGINAL_BA_FINDER_d128 MTSSL_MEGA_CrossAttention_4l_freeze_BA_FINDER_noProc_d128 \
    --baselines degree HDA CI random
```

Flags: `--budget-frac` (default 0.1 = manuscript), `--models` (default = full
d128 suite), `--baselines` (default none), `--export` (subdir under `Exports/`),
`--device cpu|cuda` (override).

**Outputs** under `Exports/<export>/`:
- `results.csv` — one row per (graph, method): ANC, start/end NPC, method name.
- `<graph>/G.el` — the processed graph.
- `<graph>/<method>_sol.txt` — ordered removed-node list (the CNDP solution).

Results are appended/checkpointed after each graph, so a long run is resumable.

### Large-graph handling — memory-scalable chunked inference (4090 / 64 GB)

Running the trained models on graphs with hundreds of thousands to millions of
nodes hits **two memory walls** that have nothing to do with model size — they
are per-node intermediates in the forward pass:

1. **Decoder outer product** `[n, d, d]` (`models.py` `test_forward`). At n=1.6M,
   d=256 this is **419 GB**. Exists in FINDER too.
2. **Cross-attention** Q/K/V `[5, n, d]` (~8 GB each transient at 1.6M).
3. **GraphSAGE message tensor** `[num_edges, d]` from PyG `propagate`. At 15.5M
   edges this is **~15 GB**, and it was what OOM-killed the first Flickr attempt
   (peak 59 GB).

All three are fixed **exactly** (bit-identical math, no quality loss) by an
additive, opt-in **chunked inference path**:

| Wall | Fix | Where (additive method) |
|---|---|---|
| decoder `[n,d,d]` | evaluate decoder in node chunks | `FINDER_DQN.test_forward_chunked` (`models.py`) |
| cross-attn `[5,n,d]` | distill in node chunks | `MEGA_Encoder.forward_chunked` (`encoders.py`) |
| SAGE `[E,d]` message | sum-agg via sparse-dense matmul (`torch.sparse.mm`) instead of PyG `propagate` | `CustomSAGEConv._forward_spmm` + `use_spmm_aggr` flag (`layers.py`) |

Driver: `Q_CNDP_Agent.calc_solution_nx_chunked` / `predict_chunked` (`Q_CNDP.py`).
None of the original methods were modified — the default forward paths are
untouched; the chunked path is selected explicitly. **Verified bit-identical**:
on Bovine/openflights/USAir97 the chunked+spmm solution equals the original
(0 nodes differ, same ANC = manuscript values).

**How the runner gates it** (`run_inference.py`):
- Auto-enabled above `CHUNK_NODE_THRESHOLD = 100000` nodes; override with
  `--chunked` / `--no-chunked`.
- `--node-chunk N` (default 20000) — node chunk size. Smaller = less peak memory.
- `--step-size K` — removals per rollout step. Auto = 1% of budget above
  `BIG_GRAPH_NODE_THRESHOLD = 20000` (fewer, larger steps → tractable wall-time on
  huge graphs), else 1.
- **GPU-first**: chunked path tries CUDA, and on CUDA OOM (e.g. the 15M-edge
  case) **auto-falls back to CPU**. For the very largest graphs (Flickr/Youtube)
  the encoder's sparse structures favor CPU/64 GB; pass `--device cpu` to skip the
  doomed GPU attempt.

NewMUSE-CNdata sizes and recommended path:

| Graph | Nodes | Edges | Recommended |
|---|---|---|---|
| Crime | 829 | 1,473 | GPU, step 1 (no chunk) |
| HI-II-14 | 4,165 | 13,087 | GPU, step 1 |
| Digg | 29,652 | 84,781 | GPU/CPU, chunked auto |
| Enron | 33,696 | 180,811 | GPU/CPU, chunked auto |
| Gnutella31 | 62,561 | 147,878 | GPU/CPU, chunked auto |
| Facebook | 63,392 | 816,831 | GPU/CPU, chunked auto |
| Epinions | 75,877 | 405,739 | chunked auto |
| Youtube | 1,134,890 | 2,987,624 | `--chunked --device cpu` |
| Flickr | 1,624,991 | 15,473,043 | `--chunked --device cpu` (heaviest) |

Flickr / MUSE-CN command (the heaviest job):

```bash
python 2026_06_21_Dissertation_context_data/run_inference.py \
    --new-graph Flickr \
    --models MTSSL_MEGA_CrossAttention_1l_freeze_BA_FINDER_noProc \
    --device cpu --node-chunk 20000 --export flickr_muse_cn
```

With the spmm aggregation the peak RAM stays well under 64 GB (vs the 59 GB
OOM-kill before the fix).

---

## 4. Experimenting on new graphs

1. Drop an edgelist (whitespace-separated `u v` per line, integer node IDs) into
   `2026_06_21_Dissertation_context_data/NewMUSE-CNdata/NewMUSE-CNdata/<Name>.txt`
   and call `--new-graph <Name>`, **or** point `--edgelist` at any path.
2. `prep_G_for_tensor` cleans it (drops isolated nodes, relabels `0..n-1`). The
   relabeling means solution node IDs in `*_sol.txt` are in that relabeled space,
   matching the saved `G.el`.
3. Budget defaults to 10% of nodes; change with `--budget-frac`.
4. No retraining is ever needed — the encoders are frozen and the agent is fully
   inductive, so the same checkpoints apply to graphs of any size/domain.

---

## 5. Model & baseline → manuscript mapping

### 5.0 Focus models (main-results comparison — what we run)

These are the six methods compared in the manuscript main-results tables
(`tab:anc_main_results`, `tab:npc_main_results`). One proposed model + one
learning baseline + four classical heuristics. **These are our focus**; the
ablation variants in §5.1 exist only to populate the ablation tables.

> **Which checkpoints?** The published tables were produced by the **non-d128**
> suite (git commit `5c08e4d "added manuscript results with ablations"`). Verified:
> running these on Bovine reproduces the table exactly — FINDER ANC = **11.88**,
> MUSE-CN ANC = **14.32**. The `_d128` dirs are a *later re-run* (embedding 128,
> May 2025) and are **not** the published numbers. **Focus on the non-d128 dirs
> below.** The runner defaults to them (`--suite manuscript`).

| Manuscript label | Type | Backing dir / heuristic |
|---|---|---|
| **MUSE-CN** | proposed | `MTSSL_MEGA_CrossAttention_1l_freeze_BA_FINDER_noProc` (5-enc, freeze, **1-layer** cross-attention) |
| **FINDER** | learning baseline | `ORIGINAL_BA_FINDER` |
| **CI** | classical baseline | `--baselines CI` — Collective Influence |
| **HDA** | classical baseline | `--baselines HDA` — High-Degree Adaptive |
| **degree** | classical baseline | `--baselines degree` — static degree centrality |
| **random** | classical baseline | `--baselines random` — random selection |

The two headline learning models (MUSE-CN + FINDER) are `--suite focus`. The full
published set (headline + ablations) is `--suite manuscript` (default).

One-shot command to reproduce the main-results comparison on a graph:

```bash
# MUSE-CN + FINDER + all four classical baselines
python 2026_06_21_Dissertation_context_data/run_inference.py --benchmark USAir97 \
    --suite focus --baselines CI HDA degree random --export main_usair97
```

### 5.1 Trained models — full suites

All live under `experiments/MEGA_integration/`. They differ only in (a) how the
5 SSL encoders are integrated and (b) the distillation module. The distinguishing
config fields are `encoder_args.encoders_list[*].load_mode` and
`encoder_args.distilation_module_args.distilation_type`.

**Published suite (`--suite manuscript`, the default).** These dirs produced the
manuscript tables (commit `5c08e4d`). All under `experiments/MEGA_integration/`.
Distinguishing config: `encoder_args.encoders_list[*].load_mode` (freeze / finetune
/ reset) and `encoder_args.distilation_module_args.distilation_type` (mlp /
cross_attention).

| Experiment dir | Encoders | Distillation | Manuscript label / role |
|---|---|---|---|
| `ORIGINAL_BA_FINDER` | single FINDER encoder | — | **FINDER** (main-results tables) |
| `MTSSL_MEGA_CrossAttention_1l_freeze_BA_FINDER_noProc` | 5 enc, `freeze` | cross-attn ×1 | **MUSE-CN** ← the headline proposed model (main-results tables) |
| `MTSSL_MEGA_reset_BA_FINDER_noProc` | 5 enc, `reset` (no pretrain) | MLP | **No pretraining** ablation = "overparameterized FINDER" (Tables anc/npc_pretraining) |
| `MTSSL_MEGA_finetune_BA_FINDER_noProc` | 5 enc, `finetune` | MLP | **MT-SSL + fine-tune** ablation (Tables anc/npc_pretraining) |
| `MTSSL_MEGA_freeze_BA_FINDER_noProc` | 5 enc, `freeze` | MLP | **MT-SSL + freeze, MLP distillation** — freeze pretraining row + **MLP baseline** of the cross-attention ablation (Tables anc/npc_CA) |
| `MTSSL_MEGA_CrossAttention_1l_finetune_BA_FINDER_noProc` | 5 enc, `finetune` | cross-attn ×1 | cross-attention + fine-tune ablation |
| `MTSSL_MEGA_CrossAttention_2l_freeze_BA_FINDER_noProc` | 5 enc, `freeze` | cross-attn ×2 | X-attn 2-layer ablation (Tables anc/npc_CA) |
| `MTSSL_MEGA_CrossAttention_4l_freeze_BA_FINDER_noProc` | 5 enc, `freeze` | cross-attn ×4 | X-attn 4-layer ablation |

Notes:
- **MUSE-CN** = MEGA 5-encoder + **frozen** pretrained weights + **1-layer
  cross-attention** distillation. The ablation finds 1-layer best; 2/4-layer dirs
  populate the layer-count ablation (Tables `anc_CA` / `npc_CA`).
- "`noProc`" = no procedural node attributes beyond the constant `ones` features
  (FINDER-style 2 input features).
- **`_d128` suite (`--suite d128`)** = a later re-run (embedding 128, May 2025) of
  the same designs (`ORIGINAL_BA_FINDER_d128`, `MTSSL_MEGA_freeze/finetune/reset
  _..._d128`, `MTSSL_MEGA_CrossAttention_{2,4,6}l_freeze_..._d128`). These are
  **not** in the published tables — use only for the newer experiments. Note the
  d128 set has no 1-layer cross-attention model.

### 5.2 The 5 SSL pretraining tasks (frozen encoders)

Loaded from `pretraining_module/SSL_models/SSL_v1_noProcAttr/seed_42_tasks_<t>/`
(the `_d128` models load from `SSL_v1_noProcAttr_d128/`).
Manuscript: "five widely used SSL tasks encompassing generative reconstruction,
whitening decorrelation, and mutual information maximization."

| Key | SSL task family |
|---|---|
| `p_recon` | feature/graph reconstruction (generative) |
| `p_link` | link prediction (generative) |
| `p_ming` | mutual information maximization, global (DGI-style) |
| `p_minsg` | mutual information maximization, subgraph/local |
| `p_decor` | whitening / feature decorrelation |

### 5.3 Classical baselines (`--baselines ...`)

From `py_modules/baselines.py` via `get_baseline_sol`. Manuscript main-results
columns:

| Runner name | Manuscript column | Description |
|---|---|---|
| `degree` | **degree** | static degree centrality, remove top-degree nodes |
| `HDA` | **HDA** | High-Degree Adaptive — recompute degree after each removal |
| `CI` | **CI** | Collective Influence algorithm |
| `random` | **random** | random node selection (lower bound) |

---

## 6. Sanity check already run

`--test --suite focus` on Bovine (121 nodes) reproduces the manuscript table
**exactly**: FINDER ANC = 11.88, MUSE-CN ANC = 14.32 (Table `anc_main_results`,
Bovine row). This confirms the non-d128 dirs are the published checkpoints.
`--new-graph Crime` (829 nodes) ran the full suite through the new-graph path
successfully. Pipeline confirmed for both existing benchmarks and new instances.
