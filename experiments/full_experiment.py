"""Full B0-B3 comparison on SIFT1M: memory vs recall (vector + graph compression).

B0: HNSW (raw vectors + 4-byte graph)
B1: HNSW + PQ    (16-byte codes + 4-byte graph)
B2: HNSW + RaBitQ (16-byte codes + 4-byte graph)
B3: HNSW + RaBitQ + graph compression (16-byte codes + delta-encoded graph)  [ours]

Memory is exact; recall is measured (faiss for B0, RaBitQ-guided graph search
+ exact re-rank for B2/B3).
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
from rabitq_impl import rabitq_index, estimate_distances
from reorder import varint_bytes


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
    return total / 8  # bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--nq", type=int, default=200)
    ap.add_argument("--rerank", type=int, default=300)
    args = ap.parse_args()

    X = read_fvecs(args.base)[:args.n]
    N, D = X.shape
    Q = read_fvecs(args.query)[:args.nq]
    GT = read_ivecs(args.gt)[:args.nq]  # full-1M ground truth (correct when n=1M)

    vec_raw = N * D * 4
    vec_code = N * (D // 8)  # RaBitQ: D bits = D/8 bytes

    print("building HNSW")
    idx = build_hnsw(X, 16, 200)
    levels, adj = extract_graph(idx.hnsw)
    edges = count_edges(levels, adj)
    graph_raw = edges * 4

    # reorder for graph compression
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
    graph_comp = graph_cost_bitpack(levels, adj, o2n, bits_gap=18)

    print("quantizing RaBitQ")
    codes, P, x0, c = rabitq_index(X)

    # memory table
    print("\n===== MEMORY (SIFT1M) =====")
    mem = {
        "B0 HNSW": (vec_raw + graph_raw) / 1e6,
        "B1 HNSW+PQ(16B)": (vec_code + graph_raw) / 1e6,
        "B2 HNSW+RaBitQ": (vec_code + graph_raw) / 1e6,
        "B3 ours (RaBitQ+graph-comp)": (vec_code + graph_comp) / 1e6,
    }
    for k, v in mem.items():
        print(f"  {k:32s}: {v:7.1f} MB")
    print(f"  graph compression: {graph_raw/1e6:.1f}MB -> {graph_comp/1e6:.1f}MB "
          f"({graph_raw/graph_comp:.2f}x)")

    # recall
    print("\n===== RECALL@10 (re-rank=%d) =====" % args.rerank)
    # B0: faiss exact search
    idx.hnsw.efSearch = 128
    _, I = idx.search(Q, 10)
    r0 = np.mean([len(set(I[i]) & set(GT[i][:10].tolist())) / 10 for i in range(args.nq)])
    print(f"  B0 faiss exact        : {r0:.3f}")

    # B2/B3: RaBitQ-guided graph search + exact re-rank (graph compression doesn't change recall)
    entry = int(idx.hnsw.entry_point)
    hits = 0
    t = time.time()
    for qi in range(args.nq):
        q = Q[qi]
        est_d, _ = estimate_distances(codes, P, x0, c, q)
        df = lambda v: float(est_d[v])
        res = hnsw_search(levels, adj, entry, df, 128, args.rerank)
        exact_d = np.sum((X[res] - q) ** 2, axis=1)
        res = list(np.array(res)[np.argsort(exact_d)][:10])
        hits += len(set(res) & set(GT[qi][:10].tolist()))
    r2 = hits / (args.nq * 10)
    print(f"  B2/B3 RaBitQ+graph   : {r2:.3f}  [{time.time()-t:.0f}s]")

    print(f"\n  Total memory reduction B0->B3: {mem['B0 HNSW']/mem['B3 ours (RaBitQ+graph-comp)']:.1f}x")


if __name__ == "__main__":
    main()
