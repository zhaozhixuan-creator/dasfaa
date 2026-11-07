"""Sweep k-NN layer-0 degree: does compression-aware construction break 2x
WITHOUT losing recall? Measures compression ratio, total memory, and recall
for k-NN layer-0 (k in {16,24,32}) + HNSW hierarchy, vs standard HNSW.
"""

import argparse
import time

import numpy as np
import faiss
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

from io_data import read_fvecs, read_ivecs
from hnsw_graph import build_hnsw, extract_graph, count_edges
from hnsw_search import hnsw_search


def golomb_rice_bytes(levels, adj, o2n, k=15):
    bits_first = int(np.ceil(np.log2(len(levels))))
    total_bits = 0
    for i in range(len(levels)):
        for l in range(int(levels[i]) + 1):
            nb = adj[i][l]
            if len(nb) == 0:
                continue
            s = np.sort(o2n[nb])
            total_bits += bits_first
            g = np.diff(s)
            total_bits += (g >> k).sum() + len(g) * (1 + k)
    return total_bits / 8


def rcm_order(levels, adj, N):
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
    return o2n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--nq", type=int, default=200)
    args = ap.parse_args()

    X = read_fvecs(args.base)[:args.n]
    N, D = X.shape
    Q = read_fvecs(args.query)[:args.nq]
    GT = read_ivecs(args.gt)[:args.nq]

    # correct subset GT
    bf = faiss.IndexFlatL2(D); bf.add(X)
    _, GT = bf.search(Q, 10)

    idx = build_hnsw(X, 16, 200)
    levels, adj = extract_graph(idx.hnsw)
    entry = int(idx.hnsw.entry_point)

    # k-NN (max k = 32)
    t = time.time()
    _, I = bf.search(X, 33)
    print(f"k-NN built [{time.time()-t:.0f}s]")

    # HNSW baseline recall (faiss)
    idx.hnsw.efSearch = 128
    _, Ih = idx.search(Q, 10)
    r_hnsw = np.mean([len(set(Ih[i]) & set(GT[i][:10].tolist())) / 10 for i in range(args.nq)])

    def recall_of(adj_layers, entry):
        hits = 0
        for qi in range(args.nq):
            q = Q[qi]
            df = lambda v: float(np.sum((X[v] - q) ** 2))
            res = hnsw_search(levels, adj_layers, entry, df, 128, 10)
            hits += len(set(res) & set(GT[qi][:10].tolist()))
        return hits / (args.nq * 10)

    # baseline HNSW
    o2n_hnsw = rcm_order(levels, adj, N)
    raw_hnsw = count_edges(levels, adj) * 4
    comp_hnsw = golomb_rice_bytes(levels, adj, o2n_hnsw)
    print(f"\n{'method':20s} {'deg':>5s} {'raw':>7s} {'comp':>7s} {'ratio':>5s} {'recall':>7s}")
    print(f"{'HNSW (faiss)':20s} {raw_hnsw/N/4:5.1f} {raw_hnsw/1e6:7.1f} {comp_hnsw/1e6:7.1f} "
          f"{raw_hnsw/comp_hnsw:5.2f} {r_hnsw:7.3f}")

    for k in (16, 24, 32):
        knn = [I[i][I[i] != i][:k] for i in range(N)]
        adj_c = [None] * N
        for i in range(N):
            li = int(levels[i])
            adj_c[i] = [knn[i]] + [adj[i][l] for l in range(1, li + 1)]
        r = recall_of(adj_c, entry)
        raw = count_edges(levels, adj_c) * 4
        o2n = rcm_order(levels, adj_c, N)
        comp = golomb_rice_bytes(levels, adj_c, o2n)
        print(f"{'k-NN k=%d + hierarchy' % k:20s} {raw/N/4:5.1f} {raw/1e6:7.1f} {comp/1e6:7.1f} "
              f"{raw/comp:5.2f} {r:7.3f}")


if __name__ == "__main__":
    main()
