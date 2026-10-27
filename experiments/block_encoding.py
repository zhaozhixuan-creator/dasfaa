"""Measure intrinsic cluster-locality of the HNSW graph, and block-encoding cost.

Key quantity: how many distinct blocks/clusters does each node's neighbor list
touch? This determines whether 2-level (block) encoding can reach 4x.

Vectorized (no per-node Python loops) so it scales to 1M.
"""

import argparse
import time

import numpy as np
import faiss
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

from io_data import read_fvecs
from hnsw_graph import build_hnsw, extract_graph, count_edges


def build_flat_edges(levels, adj):
    srcs, dsts = [], []
    for i in range(len(levels)):
        for l in range(int(levels[i]) + 1):
            nb = adj[i][l]
            if len(nb):
                srcs.append(np.full(len(nb), i, np.int64))
                dsts.append(nb.astype(np.int64))
    return np.concatenate(srcs), np.concatenate(dsts)


def distinct_per_node(src, labels, nlabel):
    """Average number of distinct label values per source node (vectorized)."""
    combined = src * nlabel + labels
    n_unique = len(np.unique(combined))
    return n_unique / src.max()  # src.max()+1 nodes (every node has >=1 edge)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--ef", type=int, default=200)
    ap.add_argument("--c", type=int, default=1024)  # k-means clusters
    args = ap.parse_args()

    X = read_fvecs(args.base)[:args.n]
    N, D = X.shape
    idx = build_hnsw(X, args.m, args.ef)
    levels, adj = extract_graph(idx.hnsw)
    edges = count_edges(levels, adj)
    print(f"N={N} edges={edges} avg_deg={edges/N:.2f}")

    src, dst = build_flat_edges(levels, adj)

    # RCM ordering -> distinct blocks (S=32/64/128)
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

    print("RCM ordering:")
    for S in (32, 64, 128):
        nblocks = (N + S - 1) // S
        k = distinct_per_node(src, o2n[dst] // S, nblocks)
        print(f"  S={S}: avg distinct blocks = {k:.1f}")

    # k-means -> intrinsic cluster locality
    t = time.time()
    km = faiss.Kmeans(D, args.c, niter=20, verbose=False, seed=0)
    km.train(X)
    _, I = km.index.search(X, 1)
    cl = I.ravel()
    k_cluster = distinct_per_node(src, cl[dst], args.c)
    print(f"k-means C={args.c}: avg distinct clusters per node = {k_cluster:.1f}  "
          f"(neighbors span {k_cluster}/{args.c} clusters)  [{time.time()-t:.0f}s]")


if __name__ == "__main__":
    main()
