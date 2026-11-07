"""Compression-aware construction: k-NN layer-0 + HNSW hierarchy (option 2).

Tests whether replacing HNSW's diversified layer-0 with a (compressible) k-NN
graph, while keeping the hierarchy for navigability, breaks 2x graph compression
without losing recall.
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


def bitpack_bytes(levels, adj, o2n, bits_gap=18):
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
    ap.add_argument("--nq", type=int, default=100)
    ap.add_argument("--knn", type=int, default=16)
    args = ap.parse_args()

    X = read_fvecs(args.base)[:args.n]
    N, D = X.shape
    Q = read_fvecs(args.query)[:args.nq]
    GT = read_ivecs(args.gt)[:args.nq]

    # correct subset ground truth
    bf = faiss.IndexFlatL2(D); bf.add(X)
    _, GT = bf.search(Q, 10)

    # HNSW (baseline + hierarchy source)
    idx = build_hnsw(X, 16, 200)
    levels, adj = extract_graph(idx.hnsw)
    entry = int(idx.hnsw.entry_point)

    # exact k-NN graph for layer-0
    t = time.time()
    _, I = bf.search(X, args.knn + 1)
    knn = [I[i][I[i] != i] for i in range(N)]
    print(f"k-NN built [{time.time()-t:.0f}s]")

    # combined adjacency: layer-0 = k-NN, upper layers = HNSW
    adj_comb = [None] * N
    for i in range(N):
        li = int(levels[i])
        layers = [knn[i]]  # layer 0 = k-NN
        for l in range(1, li + 1):
            layers.append(adj[i][l])
        adj_comb[i] = layers

    # recall of combined graph (upper hierarchy + k-NN layer-0)
    t = time.time()
    hits = 0
    for qi in range(args.nq):
        q = Q[qi]
        df = lambda v: float(np.sum((X[v] - q) ** 2))
        res = hnsw_search(levels, adj_comb, entry, df, 128, 10)
        hits += len(set(res) & set(GT[qi][:10].tolist()))
    recall_comb = hits / (args.nq * 10)
    print(f"combined (k-NN layer0 + hierarchy) recall@10 = {recall_comb:.3f}  [{time.time()-t:.0f}s]")

    # HNSW recall (faiss)
    idx.hnsw.efSearch = 128
    _, Ih = idx.search(Q, 10)
    recall_hnsw = np.mean([len(set(Ih[i]) & set(GT[i][:10].tolist())) / 10 for i in range(args.nq)])
    print(f"HNSW (faiss) recall@10 = {recall_hnsw:.3f}")

    # compression of both graphs
    o2n = rcm_order(levels, adj_comb, N)  # reorder based on combined graph
    graph_hnsw = count_edges(levels, adj) * 4
    graph_hnsw_c = bitpack_bytes(levels, adj, o2n)
    graph_comb = count_edges(levels, adj_comb) * 4
    graph_comb_c = bitpack_bytes(levels, adj_comb, o2n)
    print(f"\nHNSW graph      : {graph_hnsw/1e6:.1f}MB -> {graph_hnsw_c/1e6:.1f}MB ({graph_hnsw/graph_hnsw_c:.2f}x)")
    print(f"combined graph  : {graph_comb/1e6:.1f}MB -> {graph_comb_c/1e6:.1f}MB ({graph_comb/graph_comb_c:.2f}x)")


if __name__ == "__main__":
    main()
