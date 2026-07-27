# MUSE-CN

**MU**lti-encoder **S**elf-supervised **E**xpert for learning to identify **C**ritical
**N**odes in large graphs.

MUSE-CN is a reinforcement-learning (DQN) agent for the **Critical Node Detection
Problem (CNDP)**. It pairs a self-supervised, multi-encoder GraphSAGE backbone
(five encoders, each pretrained on a different SSL task) with a cross-attention
distillation module and a FINDER-style Q-value decoder. Nodes are removed
autoregressively by descending Q-value.

This repository is a standalone snapshot that reproduces the paper's results:
the trained models, the pretrained backbone, all 35 benchmark instances, the code
for pretraining / training / evaluation, and the recorded results.

## Layout

```
run_inference.py            # main entry: evaluate models + baselines on any graph
scalable_baselines.py       # scalable degree / HDA / CI heuristics (batched, sparse)
main.py                     # DQN training entry
Q_CNDP.py                   # the RL agent (build/train/evaluate rollout)
py_modules/                 # models, encoders, layers, utils, baselines, curriculum
cy_modules/                 # Cython graph/env/replay code (prebuilt .so for py3.9)
exp_configs/                # training experiment configs

pretraining_module/
  pretrain_model.py, pretrain_utils/, general_pretraining.sh
  SSL_models/SSL_v1_noProcAttr/     # the 5 pretrained encoders MUSE-CN loads (frozen)
Data/
  Data_benchmark/           # 31 small/medium instances (incl. PPI_1..4)
  large_graphs/             # Enron, Epinions, Youtube, Flickr (web-scale)
  SSL_data/                 # synthetic graph corpus for SSL pretraining

experiments/MEGA_integration/       # 8 trained models (best checkpoint each)
  ORIGINAL_BA_FINDER                            # FINDER (learning baseline)
  MTSSL_MEGA_CrossAttention_1l_freeze_..._noProc  # MUSE-CN (headline model)
  MTSSL_MEGA_{reset,finetune,freeze}_..._noProc   # pretraining-strategy ablations
  MTSSL_MEGA_CrossAttention_{1l_finetune,2l,4l}_..._noProc  # cross-attention ablations

results/
  large_graph_runs/         # Enron/Epinions/Youtube/Flickr runs + results.csv (models + baselines)
  benchmark_evals/          # small/medium benchmark ANC/NPC (10% budget)
  manuscript_plots/
docs/                       # method notes, scalable-inference notes, paper source
```

## Setup

Python **3.9** (the Cython extensions in `cy_modules/*.so` are built for cpython-3.9).

```bash
conda create -n muse_cn python=3.9 && conda activate muse_cn
# install a CUDA-matched PyTorch + PyTorch Geometric first (see pytorch.org / pyg docs)
pip install -r requirements.txt
```

The Cython extensions are prebuilt. If you need to rebuild them (e.g. different
Python version):

```bash
python setup.py build_ext --inplace
```

## Evaluate

`run_inference.py` loads a model (its best checkpoint), runs the autoregressive
rollout, and writes ANC / NPC to `results/runs/<export>/results.csv`.
It is crash-safe and resumable: each (model, graph) pair is checkpointed as soon
as it finishes, and re-running the same command skips finished pairs.

```bash
# the two headline models on one small benchmark instance
python run_inference.py --benchmark Bovine --suite focus --export demo

# add the classical baselines
python run_inference.py --benchmark USAir97 --suite focus \
    --baselines degree HDA CI --export demo

# a large / web-scale instance (memory-scalable chunked path auto-enables >100k nodes)
python run_inference.py --new-graph Flickr \
    --models MTSSL_MEGA_CrossAttention_1l_freeze_BA_FINDER_noProc \
    --node-chunk 8000 --export flickr

# full model suite on a benchmark graph
python run_inference.py --benchmark Openflights --suite manuscript --export demo
```

Useful flags: `--suite {focus,manuscript}`, `--models <exp dirs...>`,
`--baselines degree HDA CI`, `--budget-frac 0.1`, `--all-new-graphs`
(every graph in `Data/large_graphs`, largest→smallest), `--node-chunk`,
`--step-size`. Budget defaults to 10% of nodes.

Model → paper mapping: `ORIGINAL_BA_FINDER` = FINDER;
`MTSSL_MEGA_CrossAttention_1l_freeze_BA_FINDER_noProc` = MUSE-CN; the remaining
six are the pretraining and cross-attention ablations.

### Scalable inference / baselines

Large instances (Youtube, Flickr) do not fit a naive forward pass. The chunked
path (node-chunked decoder + cross-attention, sparse-matmul GraphSAGE
aggregation) makes them fit while staying numerically identical to the original.
`degree` / `HDA` / `CI` use batched, sparse implementations
(`scalable_baselines.py`) so they also run at web scale. Note `HDA` and `CI` here
are **adaptive** (scores are recomputed on the residual graph every `step_size`
removals). See `docs/scalable_inference.md`.

## Train

DQN training (edit the experiment list at the top of `main.py`, which reads a
config from `exp_configs/`):

```bash
python main.py
```

Each run writes to `experiments/<...>/`; evaluation picks the checkpoint with the
lowest validation connectivity.

## Pretrain the backbone

The 5 SSL encoders are already trained and included
(`pretraining_module/SSL_models/SSL_v1_noProcAttr/`). To retrain from the
synthetic corpus (`Data/SSL_data/`):

```bash
cd pretraining_module
bash general_pretraining.sh
```

## Reproducing the numbers

`results/large_graph_runs/results.csv` holds the recorded ANC values for the
web-scale instances (models + baselines); `results/benchmark_evals/` holds the
small/medium benchmark results. Example: Bovine MUSE-CN ANC = 14.32,
Flickr MUSE-CN 32.63 vs FINDER 37.35.
