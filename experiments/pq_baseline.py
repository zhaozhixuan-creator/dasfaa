"""PQ baseline (B1): HNSW graph + PQ distances (16-byte codes) + exact re-rank.

Compares against RaBitQ (same memory: 16 bytes/vector) to validate the
"RaBitQ > PQ for graph search" claim.
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
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--nq", type=int, default=200)
    ap.add_argument("--rerank", type=int, default=2000)
    args = ap.parse_args()

    X = read_fvecs(args.base)[:args.n]
    N, D = X.shape
    Q = read_fvecs(args.query)[:args.nq]
    GT = read_ivecs(args.gt)[:args.nq]

    idx = build_hnsw(X, 16, 200)
    levels, adj = extract_graph(idx.hnsw)
    entry = int(idx.hnsw.entry_point)

    # PQ: 16 bytes/vector (M=16 subquantizers, 8-bit each)
    print("training PQ (M=16, 8-bit = 16B/vector)")
    t = time.time()
    pq = faiss.ProductQuantizer(D, 16, 8)
    pq.train(X[: min(N, 100_000)])  # train on subset for speed
    codes = pq.compute_codes(X)
    print(f"PQ trained+encoded [{time.time()-t:.0f}s]")

    # RaBitQ for comparison
    print("quantizing RaBitQ")
    rcodes, P, x0, c = rabitq_index(X)

    for mode in ("pq", "rabitq"):
        hits = 0
        t = time.time()
        for qi in range(args.nq):
            q = Q[qi]
            if mode == "pq":
                dsub = D // 16
                q_sub = q.reshape(16, dsub).astype(np.float32)
                cent = faiss.vector_to_array(pq.centroids).reshape(16, 256, dsub)
                dt = ((cent - q_sub[:, None, :]) ** 2).sum(axis=2)  # 16 x 256
                pq_dist = dt[np.arange(16)[None, :], codes].sum(axis=1)
                df = lambda v: float(pq_dist[v])
            else:
                est_d, _ = estimate_distances(rcodes, P, x0, c, q)
                df = lambda v: float(est_d[v])
            res = hnsw_search(levels, adj, entry, df, 512, args.rerank)
            exact_d = np.sum((X[res] - q) ** 2, axis=1)
            res = list(np.array(res)[np.argsort(exact_d)][:10])
            hits += len(set(res) & set(GT[qi][:10].tolist()))
        recall = hits / (args.nq * 10)
        print(f"  {mode:7s}: recall@10 = {recall:.3f}  [{time.time()-t:.0f}s]")


if __name__ == "__main__":
    main()
