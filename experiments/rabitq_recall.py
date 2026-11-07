"""Validate: does RaBitQ-guided graph search achieve high recall (close to exact)?

Builds HNSW (faiss), extracts the graph, then runs the same beam search on
layer 0 using exact distances vs RaBitQ-estimated distances (+ exact re-rank).
"""

import argparse
import time

import numpy as np
import faiss

from io_data import read_fvecs, read_ivecs
from hnsw_graph import build_hnsw, extract_graph
from hnsw_search import hnsw_search
from rabitq_impl import rabitq_index, estimate_distances


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--nq", type=int, default=100)
    ap.add_argument("--ef", type=int, default=64)
    args = ap.parse_args()

    X = read_fvecs(args.base)[:args.n]
    N, D = X.shape
    Q = read_fvecs(args.query)[:args.nq]
    GT = read_ivecs(args.gt)[:args.nq]

    print("building HNSW")
    idx = build_hnsw(X, 16, 200)
    levels, adj = extract_graph(idx.hnsw)
    entry = int(idx.hnsw.entry_point)

    print("quantizing with RaBitQ")
    codes, P, x0, c = rabitq_index(X)

    for mode in ("exact", "rabitq"):
        hits = 0
        t = time.time()
        for qi in range(args.nq):
            q = Q[qi]
            if mode == "exact":
                def df(v):
                    return float(np.sum((X[v] - q) ** 2))
            else:
                est_d, _ = estimate_distances(codes, P, x0, c, q)
                def df(v):
                    return float(est_d[v])
            res = hnsw_search(levels, adj, entry, df, args.ef, 10)
            # exact re-rank of the returned candidates
            exact_d = np.sum((X[res] - q) ** 2, axis=1)
            res = list(np.array(res)[np.argsort(exact_d)][:10])
            hits += len(set(res) & set(GT[qi][:10].tolist()))
        recall = hits / (args.nq * 10)
        print(f"  {mode:6s}: recall@10 = {recall:.3f}  [{time.time()-t:.0f}s]")


if __name__ == "__main__":
    main()
