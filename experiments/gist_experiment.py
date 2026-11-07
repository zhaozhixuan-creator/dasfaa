"""GIST1M (1M x 960) B0-B3 robustness experiment.

The GIST tar lacks a query set, so we sample queries from the base and compute
ground truth by brute force.
"""

import argparse
import time

import numpy as np
import faiss
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

from io_data import read_fvecs
from hnsw_graph import build_hnsw, extract_graph, count_edges
from hnsw_search import hnsw_search
from rabitq_impl import rabitq_index, estimate_distances


def graph_cost_bitpack(levels, adj, o2n, bits_gap=18):
    bits_first = int(np.ceil(np.log2(len(levels))))
    total = 0
    for i in range(len(levels)):
        for l in range(int(levels[i]) + 1):
            nb = adj[i][l]
            if len(nb) == 0:
                continue
            s = np.sort(o2n[nb])
            total += bits_first
            gaps = np.diff(s)
            total += len(gaps) * bits_gap + int(np.sum(gaps >= (1 << bits_gap))) * 32
    return total / 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--nq", type=int, default=200)
    ap.add_argument("--rerank", type=int, default=2000)
    args = ap.parse_args()

    X = read_fvecs(args.base)[:args.n]
    N, D = X.shape
    rng = np.random.default_rng(0)
    qidx = rng.choice(N, size=args.nq, replace=False)
    Q = X[qidx]

    # ground truth via brute force (excluding self)
    bf = faiss.IndexFlatL2(D)
    bf.add(X)
    _, GT = bf.search(Q, 11)  # top-11 (self + 10)
    GT = [GT[i][GT[i] != qidx[i]][:10] for i in range(args.nq)]

    vec_raw = N * D * 4
    vec_code = N * (D // 8)

    print(f"building HNSW on GIST (N={N}, D={D})")
    idx = build_hnsw(X, 16, 200)
    levels, adj = extract_graph(idx.hnsw)
    edges = count_edges(levels, adj)
    graph_raw = edges * 4

    rows, cols = [], []
    for i in range(N):
        for l in range(int(levels[i]) + 1):
            nb = adj[i][l]
            if len(nb):
                rows.append(np.full(len(nb), i, np.int32))
                cols.append(nb.astype(np.int32))
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    A = csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(N, N))
    A = (A + A.T).tocsr(); A.data[:] = 1
    perm = reverse_cuthill_mckee(A, symmetric_mode=True).astype(np.int64)
    o2n = np.empty(N, dtype=np.int64); o2n[perm] = np.arange(N)
    graph_comp = graph_cost_bitpack(levels, adj, o2n)

    print("quantizing RaBitQ (960-dim)")
    codes, P, x0, c = rabitq_index(X)

    print("\n===== GIST MEMORY =====")
    print(f"  B0 HNSW          : {(vec_raw + graph_raw)/1e6:7.1f} MB")
    print(f"  B2 HNSW+RaBitQ   : {(vec_code + graph_raw)/1e6:7.1f} MB")
    print(f"  B3 ours          : {(vec_code + graph_comp)/1e6:7.1f} MB")
    print(f"  graph compression: {graph_raw/1e6:.1f}MB -> {graph_comp/1e6:.1f}MB ({graph_raw/graph_comp:.2f}x)")
    print(f"  total B0->B3     : {(vec_raw+graph_raw)/(vec_code+graph_comp):.1f}x")

    # recall
    print("\n===== GIST RECALL@10 (rerank=%d) =====" % args.rerank)
    idx.hnsw.efSearch = 128
    _, Ih = idx.search(Q, 10)
    r0 = np.mean([len(set(Ih[i]) & set(GT[i].tolist())) / 10 for i in range(args.nq)])
    print(f"  B0 faiss exact    : {r0:.3f}")

    entry = int(idx.hnsw.entry_point)
    hits = 0
    t = time.time()
    for qi in range(args.nq):
        q = Q[qi]
        est_d, _ = estimate_distances(codes, P, x0, c, q)
        df = lambda v: float(est_d[v])
        res = hnsw_search(levels, adj, entry, df, 512, args.rerank)
        exact_d = np.sum((X[res] - q) ** 2, axis=1)
        res = list(np.array(res)[np.argsort(exact_d)][:10])
        hits += len(set(res) & set(GT[qi].tolist()))
    r2 = hits / (args.nq * 10)
    print(f"  B2/B3 RaBitQ+graph: {r2:.3f}  [{time.time()-t:.0f}s]")


if __name__ == "__main__":
    main()
