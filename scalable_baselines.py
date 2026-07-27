

import heapq

import numpy as np
import networkx as nx
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# HDA — batched High-Degree-Adaptive
# ---------------------------------------------------------------------------
def scalable_HDA(G, budget, step_size, print_progress=False):

    budget = int(budget)
    step_size = max(1, int(step_size))

    degrees = dict(G.degree())
    alive = set(G.nodes())
    removed = []

    n_rounds = 0
    while len(removed) < budget and alive:
        k = min(step_size, budget - len(removed))
        # top-k alive nodes by current (residual) degree
        batch = heapq.nlargest(k, alive, key=degrees.__getitem__)

        # no edges left among alive nodes -> rest are degree 0, fill arbitrarily
        if degrees[batch[0]] <= 0:
            need = budget - len(removed)
            removed.extend(list(alive)[:need])
            break

        for v in batch:
            alive.discard(v)
        removed.extend(batch)

        # residual degree update for surviving neighbors
        for v in batch:
            for nb in G.neighbors(v):
                if nb in alive and degrees[nb] > 0:
                    degrees[nb] -= 1

        n_rounds += 1
        if print_progress and (n_rounds % 25 == 0 or len(removed) >= budget):
            print(f"  [HDA] round {n_rounds}: removed {len(removed)}/{budget}",
                  flush=True)

    return removed[:budget]


# ---------------------------------------------------------------------------
# CI — batched Collective Influence (sparse, row-chunked)
# ---------------------------------------------------------------------------
def _ci_scores_sparse(A, radius, row_chunk):

    A = A.tocsr()
    m = A.shape[0]
    deg = np.asarray(A.sum(axis=1)).ravel()
    w = deg - 1.0                      # (k_j - 1) weight vector
    scores = np.empty(m, dtype=np.float64)

    for s in range(0, m, row_chunk):
        e = min(s + row_chunk, m)
        # (A**radius) rows for this block: A[s:e] @ A @ ... (radius-1 times)
        blk = A[s:e]
        for _ in range(radius - 1):
            blk = blk @ A
        boundary = (blk > 0)           # boolean reachability, dedup per (i,j)
        boundary_sum = boundary.dot(w)  # sum of (k_j - 1) over the boundary
        scores[s:e] = w[s:e] * np.asarray(boundary_sum).ravel()

    return scores


def scalable_CI(G, budget, step_size, radius=2, node_chunk_size=8000,
                print_progress=False):

    budget = int(budget)
    step_size = max(1, int(step_size))
    row_chunk = max(1, int(node_chunk_size))

    alive = list(G.nodes())            # stable ordering of survivors
    alive_set = set(alive)
    removed = []

    n_rounds = 0
    while len(removed) < budget and alive_set:
        nodes = [v for v in alive if v in alive_set]
        alive = nodes                  # compact the ordering
        m = len(nodes)
        if m == 0:
            break

        # CSR adjacency of the residual subgraph (binary, symmetric, no dense)
        A = nx.to_scipy_sparse_array(
            G.subgraph(nodes), nodelist=nodes, format="csr", dtype=np.float64
        )
        A.data[:] = 1.0                # ensure binary

        scores = _ci_scores_sparse(A, radius, row_chunk)

        k = min(step_size, budget - len(removed))

        # all scores zero -> no boundary structure left; fill the rest by degree
        if scores.max() <= 0:
            deg = np.asarray(A.sum(axis=1)).ravel()
            order = np.argsort(deg)[::-1]
            need = budget - len(removed)
            removed.extend(nodes[i] for i in order[:need])
            break

        # top-k by score (partial sort, then order the k descending)
        top = np.argpartition(scores, m - k)[m - k:]
        top = top[np.argsort(scores[top])[::-1]]
        batch = [nodes[i] for i in top]

        for v in batch:
            alive_set.discard(v)
        removed.extend(batch)

        n_rounds += 1
        if print_progress:
            print(f"  [CI] round {n_rounds}: removed {len(removed)}/{budget} "
                  f"(alive {len(alive_set)})", flush=True)

    return removed[:budget]
