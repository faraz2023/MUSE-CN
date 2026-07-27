

import os
import sys
import gc
import json
import argparse

import networkx as nx
import pandas as pd
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = THIS_DIR
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from py_modules.baselines import get_baseline_sol            # noqa: E402
from scalable_baselines import scalable_HDA, scalable_CI     # noqa: E402
from py_modules.utils import calc_ANC_score, calc_pairwise_connectivity  # noqa: E402
from py_modules.morPOP_utils import prep_G_for_tensor        # noqa: E402
from Q_CNDP import Q_CNDP_Agent                              # noqa: E402

FLOAT_DTYPE = torch.float32

MODEL_SUITE_MANUSCRIPT = [
    "ORIGINAL_BA_FINDER",                                # FINDER baseline
    "MTSSL_MEGA_CrossAttention_1l_freeze_BA_FINDER_noProc",   # MUSE-CN (headline)
    "MTSSL_MEGA_reset_BA_FINDER_noProc",                 # no-pretrain (overparam FINDER)
    "MTSSL_MEGA_finetune_BA_FINDER_noProc",              # MEGA + finetune (MLP distill)
    "MTSSL_MEGA_freeze_BA_FINDER_noProc",                # MEGA + freeze  (MLP distill / X-attn MLP baseline)
    "MTSSL_MEGA_CrossAttention_1l_finetune_BA_FINDER_noProc", # X-attn finetune ablation
    "MTSSL_MEGA_CrossAttention_2l_freeze_BA_FINDER_noProc",   # X-attn 2-layer ablation
    "MTSSL_MEGA_CrossAttention_4l_freeze_BA_FINDER_noProc",   # X-attn 4-layer ablation
]

MODEL_SUITE_FOCUS = [
    "ORIGINAL_BA_FINDER",                                # FINDER
    "MTSSL_MEGA_CrossAttention_1l_freeze_BA_FINDER_noProc",   # MUSE-CN
]

MODEL_SUITE_D128 = [
    "ORIGINAL_BA_FINDER_d128",
    "MTSSL_MEGA_reset_BA_FINDER_noProc_d128",
    "MTSSL_MEGA_finetune_BA_FINDER_noProc_d128",
    "MTSSL_MEGA_freeze_BA_FINDER_noProc_d128",
    "MTSSL_MEGA_CrossAttention_2l_freeze_BA_FINDER_noProc_d128",
    "MTSSL_MEGA_CrossAttention_4l_freeze_BA_FINDER_noProc_d128",
    "MTSSL_MEGA_CrossAttention_6l_freeze_BA_FINDER_noProc_d128",
]

SUITES = {
    "manuscript": MODEL_SUITE_MANUSCRIPT,
    "focus": MODEL_SUITE_FOCUS,
    "d128": MODEL_SUITE_D128,
}

EXPERIMENTS_ROOT = os.path.join("experiments", "MEGA_integration")
BENCHMARK_DIR = os.path.join("Data", "Data_benchmark")
NEW_GRAPH_DIR = os.path.join("Data", "large_graphs")

NEW_GRAPH_NODE_COUNTS = {
    "Flickr": 1624991, "Youtube": 1134890, "Epinions": 75877, "Enron": 33696,
}
DEFAULT_EXPORT_ROOT = os.path.join("results", "runs")

# Above this many nodes, use chunked step-size (mirrors eval_test.py).
BIG_GRAPH_NODE_THRESHOLD = 20000
CHUNK_NODE_THRESHOLD = 100000


def find_best_model_path(exp_path):

    levels_dir = os.path.join(exp_path, "levels")
    best_model_path = None
    best_vc = float("inf")
    for level_dir in os.listdir(levels_dir):
        level_log_path = os.path.join(levels_dir, level_dir, "level_log.json")
        with open(level_log_path, "r") as f:
            level_log = json.load(f)
        for entry in level_log["checkpoint_vcs"]:
            if entry["vc"] < best_vc:
                best_vc = entry["vc"]
                best_model_path = os.path.join(
                    levels_dir, level_dir, "models",
                    f"model_iter_{entry['iteration']}.ckpt",
                )
    return best_model_path


def load_graph(spec):

    label = spec["G_label"]
    G = spec.get("G")
    if G is None:
        G = prep_G_for_tensor(spec["path"])
    n = G.number_of_nodes()
    budget = spec.get("budget")
    if budget is None:
        budget = int(n * spec.get("budget_frac", 0.1))
    return G, label, int(budget)


def run_one_model(exp_name, G, budget, G_dir, G_pc, device_override=None,
                  chunked="auto", node_chunk_size=20000, step_size_override=None):

    exp_path = os.path.join(EXPERIMENTS_ROOT, exp_name)
    best_model_path = find_best_model_path(exp_path)

    with open(os.path.join(exp_path, "params.json"), "r") as f:
        params = json.load(f)
    params["dtype"] = FLOAT_DTYPE

    n = G.number_of_nodes()
    use_chunked = (chunked is True) or (chunked == "auto" and n > CHUNK_NODE_THRESHOLD)

    # step size: explicit override, else 1% of budget for big graphs (time), else 1
    if step_size_override is not None:
        step_size = max(1, step_size_override)
    elif n > BIG_GRAPH_NODE_THRESHOLD:
        step_size = max(1, int(budget * 0.01))
    else:
        step_size = 1

    # device: chunked path is GPU-first (fits the 4090 after chunking). Only the
    # legacy non-chunked path falls back to CPU for very large graphs.
    if device_override is not None:
        params["device"] = device_override
    elif use_chunked:
        params["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    elif n > BIG_GRAPH_NODE_THRESHOLD:
        params["device"] = "cpu"

    print(f"\n=== [{exp_name}] device={params['device']} step_size={step_size} "
          f"chunked={use_chunked}"
          f"{' chunk='+str(node_chunk_size) if use_chunked else ''} ===")
    def _build_and_run(p):
        agent = Q_CNDP_Agent(p)
        print(f"Loading model from {best_model_path}")
        agent.LoadModel(best_model_path)
        if use_chunked:
            sol = agent.calc_solution_nx_chunked(
                G, budget, step_size=step_size, node_chunk_size=node_chunk_size,
                print_progress=True)
        else:
            sol = agent.calc_solution_nx(
                G, budget, step_size=step_size, print_progress=True)
        return agent, sol

    try:
        agent, removed_nodes = _build_and_run(params)
    except RuntimeError as e:
        # GPU-first: on CUDA OOM (e.g. 15M-edge message tensor on the encoder),
        # fall back to CPU/64GB RAM. The chunked decoder keeps RAM bounded there.
        if "out of memory" in str(e).lower() and params.get("device") != "cpu":
            print(f"  !! CUDA OOM on {exp_name}; retrying on CPU. ({e})")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            params["device"] = "cpu"
            agent, removed_nodes = _build_and_run(params)
        else:
            raise

    ANC_score, _ = calc_ANC_score(
        G, removed_nodes, return_scores=True, normalize=True, step_size=step_size
    )
    G_removed = G.copy()
    G_removed.remove_nodes_from(removed_nodes)
    end_pc = calc_pairwise_connectivity(
        G_removed, normalize=True, total_nodes=G.number_of_nodes()
    )

    with open(os.path.join(G_dir, f"{exp_name}_sol.txt"), "w") as f:
        for node in removed_nodes:
            f.write(f"{node}\n")

    del agent
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "G_label": os.path.basename(G_dir),
        "ANC": ANC_score,
        "start_pairwise_connectivity": G_pc,
        "end_pairwise_connectivity": end_pc,
        "baseline": False,
        "method_name": exp_name,
    }


def run_one_baseline(baseline, G, budget, G_dir, G_pc,
                     node_chunk_size=20000, step_size_override=None):
    """Run a classical heuristic baseline (random/degree/HDA/CI).

    HDA and CI use the scalable, *batched* implementations from
    ``scalable_baselines`` (residual-degree HDA; sparse, row-chunked CI). They
    remove ``step_size`` nodes per recompute — the same step size the learning
    models use — so the big graphs that crashed the dense/serial versions now run.
    random and degree are unchanged (already scale).
    """
    print(f"\n=== [baseline:{baseline}] ===")
    # step size = removals per recompute (mirrors the learning-model rollout)
    if step_size_override is not None:
        step_size = max(1, step_size_override)
    elif G.number_of_nodes() > BIG_GRAPH_NODE_THRESHOLD:
        step_size = max(1, int(budget * 0.01))
    else:
        step_size = 1

    if baseline == 'HDA':
        print(f"  [scalable HDA] step_size={step_size}", flush=True)
        removed_nodes = scalable_HDA(G, budget, step_size, print_progress=True)
    elif baseline == 'CI':
        print(f"  [scalable CI] step_size={step_size} "
              f"row_chunk={node_chunk_size} radius=2", flush=True)
        removed_nodes = scalable_CI(G, budget, step_size, radius=2,
                                    node_chunk_size=node_chunk_size,
                                    print_progress=True)
    else:
        removed_nodes = get_baseline_sol(G, budget, baseline)
    ANC_score, _ = calc_ANC_score(
        G, removed_nodes, return_scores=True, normalize=True, step_size=step_size
    )
    G_removed = G.copy()
    G_removed.remove_nodes_from(removed_nodes)
    end_pc = calc_pairwise_connectivity(
        G_removed, normalize=True, total_nodes=G.number_of_nodes()
    )
    with open(os.path.join(G_dir, f"{baseline}_sol.txt"), "w") as f:
        for node in removed_nodes:
            f.write(f"{node}\n")
    return {
        "G_label": os.path.basename(G_dir),
        "ANC": ANC_score,
        "start_pairwise_connectivity": G_pc,
        "end_pairwise_connectivity": end_pc,
        "baseline": True,
        "method_name": baseline,
    }


def evaluate(graph_specs, model_names, baselines, export_path, device_override=None,
             chunked="auto", node_chunk_size=20000, step_size_override=None):
    os.makedirs(export_path, exist_ok=True)
    df_path = os.path.join(export_path, "results.csv")
    results = []
    if os.path.exists(df_path):
        results = pd.read_csv(df_path).to_dict(orient="records")
        print(f"Loaded {len(results)} existing rows from {df_path}")

    # Resume set: (G_label, method_name) pairs already in results.csv are skipped.
    done = set((str(r["G_label"]), str(r["method_name"])) for r in results)

    def checkpoint():
        # write to a temp file then atomically replace, so a power loss mid-write
        # cannot corrupt results.csv
        tmp = df_path + ".tmp"
        pd.DataFrame(results).to_csv(tmp, index=False)
        os.replace(tmp, df_path)

    for spec in graph_specs:
        label = spec["G_label"]
        # Decide if all pairs for this graph are already done -> skip loading it.
        pending_models = [m for m in model_names if (label, m) not in done]
        pending_bl = [b for b in baselines if (label, b) not in done]
        if not pending_models and not pending_bl:
            print(f"\n##### Graph {label}: all pairs done, skipping.")
            continue

        G, _, budget = load_graph(spec)
        G_dir = os.path.join(export_path, label)
        os.makedirs(G_dir, exist_ok=True)
        gel_path = os.path.join(G_dir, "G.el")
        if not os.path.exists(gel_path):
            nx.write_edgelist(G, gel_path)
        G_pc = calc_pairwise_connectivity(
            G, normalize=True, total_nodes=G.number_of_nodes()
        )
        print(f"\n##### Graph {label}: {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges, budget {budget}")

        for exp_name in model_names:
            if (label, exp_name) in done:
                print(f"  .. {exp_name}: already done, skipping.")
                continue
            gc.collect()
            try:
                row = run_one_model(exp_name, G, budget, G_dir, G_pc, device_override,
                                    chunked=chunked, node_chunk_size=node_chunk_size,
                                    step_size_override=step_size_override)
                results.append(row)
                done.add((label, exp_name))
                checkpoint()  # <-- immediate export after EACH (model, graph) pair
                print(f"  -> {exp_name}: ANC={row['ANC']:.4f}  "
                      f"end_PC={row['end_pairwise_connectivity']:.4f}  [saved]")
            except Exception as e:
                print(f"  !! {exp_name} FAILED: {e}")
                import traceback
                traceback.print_exc()

        for baseline in baselines:
            if (label, baseline) in done:
                print(f"  .. {baseline}: already done, skipping.")
                continue
            gc.collect()
            try:
                row = run_one_baseline(baseline, G, budget, G_dir, G_pc,
                                       node_chunk_size=node_chunk_size,
                                       step_size_override=step_size_override)
                results.append(row)
                done.add((label, baseline))
                checkpoint()
                print(f"  -> {baseline}: ANC={row['ANC']:.4f}  [saved]")
            except Exception as e:
                print(f"  !! baseline {baseline} FAILED: {e}")
                import traceback
                traceback.print_exc()

        # free this graph before loading the next (critical for the giant graphs)
        del G
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


def build_graph_specs(args):
    specs = []
    if args.test:
        # smallest existing benchmark instance -> fast smoke test
        G = prep_G_for_tensor(os.path.join(BENCHMARK_DIR, "Bovine.txt"))
        specs.append({"G": G, "G_label": "Bovine", "budget_frac": args.budget_frac})
        return specs

    if args.benchmark:
        path = os.path.join(BENCHMARK_DIR, f"{args.benchmark}.txt")
        G = prep_G_for_tensor(path)
        specs.append({"G": G, "G_label": args.benchmark, "budget_frac": args.budget_frac})

    if args.new_graph:
        path = os.path.join(NEW_GRAPH_DIR, f"{args.new_graph}.txt")
        G = prep_G_for_tensor(path)
        specs.append({"G": G, "G_label": args.new_graph, "budget_frac": args.budget_frac})

    if args.all_new_graphs:
        # every .txt in NewMUSE-CNdata, ordered largest -> smallest by node count.
        # Lazy ("path") specs so only one graph is held in memory at a time.
        files = [f for f in os.listdir(NEW_GRAPH_DIR) if f.endswith(".txt")]

        def _size_key(fname):
            name = fname[:-4]
            n = NEW_GRAPH_NODE_COUNTS.get(name)
            if n is not None:
                return (1, n)  # known node count ranks above unknown
            return (0, os.path.getsize(os.path.join(NEW_GRAPH_DIR, fname)))

        files.sort(key=_size_key, reverse=True)  # largest first
        for f in files:
            specs.append({"path": os.path.join(NEW_GRAPH_DIR, f),
                          "G_label": f[:-4], "budget_frac": args.budget_frac})

    if args.edgelist:
        G = prep_G_for_tensor(args.edgelist)
        label = args.label or os.path.splitext(os.path.basename(args.edgelist))[0]
        specs.append({"G": G, "G_label": label, "budget_frac": args.budget_frac})

    if not specs:
        raise SystemExit(
            "No graph selected. Use --test, --benchmark, --new-graph, or --edgelist."
        )
    return specs


def main():
    p = argparse.ArgumentParser(description="MUSE-CN additive inference runner")
    src = p.add_argument_group("graph source (choose one or more)")
    src.add_argument("--test", action="store_true",
                     help="smoke test: all models on Bovine, export to Exports/test")
    src.add_argument("--benchmark", help="existing benchmark name, e.g. USAir97")
    src.add_argument("--new-graph", help="NewMUSE-CNdata name, e.g. Crime / Enron")
    src.add_argument("--all-new-graphs", action="store_true",
                     help="run every graph in NewMUSE-CNdata, largest->smallest (lazy load)")
    src.add_argument("--edgelist", help="path to an arbitrary edgelist file")
    src.add_argument("--label", help="label for --edgelist graph")

    p.add_argument("--budget-frac", type=float, default=0.1,
                   help="budget as fraction of node count (default 0.1, manuscript)")
    p.add_argument("--suite", default="manuscript", choices=list(SUITES.keys()),
                   help="model suite: manuscript (default, published), focus (2 headline), d128 (re-run)")
    p.add_argument("--models", nargs="*", default=None,
                   help="explicit experiment dirs to run (overrides --suite)")
    p.add_argument("--baselines", nargs="*", default=[],
                   help="heuristics: random degree HDA CI")
    p.add_argument("--export", default=None,
                   help="export subdir under Exports/ (default: 'test' if --test else 'run')")
    p.add_argument("--device", default=None, choices=[None, "cpu", "cuda"],
                   help="override device for all models")
    p.add_argument("--chunked", dest="chunked", action="store_true", default=None,
                   help="force memory-scalable chunked inference (default: auto >100k nodes)")
    p.add_argument("--no-chunked", dest="chunked", action="store_false",
                   help="force-disable chunked inference")
    p.add_argument("--node-chunk", type=int, default=20000,
                   help="node chunk size for chunked inference (default 20000)")
    p.add_argument("--step-size", type=int, default=None,
                   help="override removals per rollout step (default: auto, 1%% budget for big graphs)")
    args = p.parse_args()

    chunked = "auto" if args.chunked is None else args.chunked

    model_names = args.models if args.models is not None else SUITES[args.suite]
    export_name = args.export or ("test" if args.test else "run")
    export_path = os.path.join(DEFAULT_EXPORT_ROOT, export_name)

    specs = build_graph_specs(args)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Export path: {export_path}")
    print(f"Models: {model_names}")

    evaluate(specs, model_names, args.baselines, export_path, args.device,
             chunked=chunked, node_chunk_size=args.node_chunk,
             step_size_override=args.step_size)
    print("\nDONE.")


if __name__ == "__main__":
    main()
