"""Verify the core hypothesis behind option 2:

  reducing diversification (exact k-NN graph) makes the graph much more
  compressible, at some recall cost.

Compares, on the same SIFT subset:
  - exact k-NN graph (k=32): the "fully local" graph, no diversification
  - HNSW layer-0 graph: the diversified graph used in practice

For both we measure (a) varint+delta compression after RCM reordering, and
(b) recall@10 under the *same* beam search over layer 0, so the comparison
isolates the effect of graph structure (diversification).
"""

import argparse
import heapq
import time

import numpy as np
import faiss
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

from io_data import read_fvecs, read_ivecs
from hnsw_graph import build_hnsw, extract_graph
from reorder import varint_bytes


def rcm_order_from_adj(adj_lists, N):
    rows, cols = [], []
    for i, nb in enumerate(adj_lists):
        if len(nb):
            rows.append(np.full(len(nb), i, np.int32))
            cols.append(nb.astype(np.int32))
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    A = csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(N, N))
    A = (A + A.T).tocsr(); A.data[:] = 1
    perm = reverse_cuthill_mckee(A, symmetric_mode=True).astype(np.int64)
    o2n = np.empty(N, dtype=np.int64); o2n[perm] = np.arange(N)
    return o2n


def compress(adj_lists, o2n):
    orig = 0; cost = 0
    for nb in adj_lists:
        orig += len(nb) * 4
        if len(nb) == 0:
            continue
        s = np.sort(o2n[nb])
        cost += varint_bytes(int(s[0]))
        for x in np.diff(s):
            cost += varint_bytes(int(x))
    return orig, cost


def beam_search(adj_lists, X, q, k, budget, nstarts, rng):
    """Beam search over a generic (layer-0) adjacency. Returns k nearest ids.

    `budget` caps the number of node expansions (distance computations), so both
    graphs are compared at equal search cost.
    """
    N = X.shape[0]
    starts = rng.choice(N, size=nstarts, replace=False)
    visited = set()
    candidates = []  # min-heap of (dist, id)
    best = []        # max-heap of (-dist, id), size k
    for s in starts:
        heapq.heappush(candidates, (float(np.sum((X[s] - q) ** 2)), int(s)))
    expanded = 0
    while candidates and expanded < budget:
        d, u = heapq.heappop(candidates)
        if u in visited:
            continue
        visited.add(u)
        expanded += 1
        heapq.heappush(best, (-d, u))
        if len(best) > k:
            heapq.heappop(best)
        for v in adj_lists[u]:
            v = int(v)
            if v not in visited:
                heapq.heappush(candidates, (float(np.sum((X[v] - q) ** 2)), v))
    return [x[1] for x in sorted(best, key=lambda t: -t[0])][:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--knn", type=int, default=32)
    ap.add_argument("--m", type=int, default=16)
    args = ap.parse_args()

    X = read_fvecs(args.base)[:args.n]
    N, D = X.shape
    Q = read_fvecs(args.query)[:args.nq]
    GT = read_ivecs(args.gt)[:args.nq]

    print(f"N={N} D={D} nq={args.nq}")

    # exact k-NN graph
    t = time.time()
    bf = faiss.IndexFlatL2(D)
    bf.add(X)
    _, I = bf.search(X, args.knn + 1)
    knn_adj = [I[i][I[i] != i] for i in range(N)]
    print(f"exact k-NN built [{time.time()-t:.0f}s]")

    # HNSW layer-0
    idx = build_hnsw(X, args.m, 200)
    levels, adj = extract_graph(idx.hnsw)
    hnsw_adj = [adj[i][0] for i in range(N)]

    for name, adj_lists in (("exact-kNN", knn_adj), ("HNSW", hnsw_adj)):
        o2n = rcm_order_from_adj(adj_lists, N)
        orig, cost = compress(adj_lists, o2n)
        deg = np.mean([len(nb) for nb in adj_lists])

        rng = np.random.default_rng(0)
        t = time.time()
        hits = 0
        for qi in range(args.nq):
            res = beam_search(adj_lists, X, Q[qi], 10, budget=500, nstarts=8, rng=rng)
            hits += len(set(res) & set(GT[qi][:10].tolist()))
        recall = hits / (args.nq * 10)
        print(f"  {name:10s}: avg_deg={deg:.1f}  compressed={cost/1e6:5.1f}MB "
              f"({orig/cost:5.2f}x vs {orig/1e6:.1f}MB)  recall@10={recall:.3f}  "
              f"[search {time.time()-t:.0f}s]")


if __name__ == "__main__":
    main()
