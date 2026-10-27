"""Run graph compression (reorder + delta) on SIFT1M and report the ratio.

Usage:
    ../.venv/bin/python sift_experiment.py --base ../data/sift/sift_base.fvecs \
        --query ../data/sift/sift_query.fvecs --gt ../data/sift/sift_groundtruth.ivecs
"""

import argparse
import time

import numpy as np

from io_data import read_fvecs, read_ivecs
from hnsw_graph import build_hnsw, extract_graph, count_edges
from reorder import symmetrized_adjacency, rcm_order, delta_encode, original_bytes


def morton_order(X, bits=14):
    N, D = X.shape
    Xq = ((X - X.min(0)) / (X.max(0) - X.min(0) + 1e-12) * ((1 << bits) - 1)).astype(np.int64)
    code = np.zeros(N, dtype=np.int64)
    for b in reversed(range(bits)):  # MSB first -> proper z-order
        for dd in range(D):
            code = (code << 1) | ((Xq[:, dd] >> b) & 1)
    o2n = np.empty(N, dtype=np.int64)
    o2n[np.argsort(code)] = np.arange(N)
    return o2n


def gap_hist(levels, adj, o2n):
    g = []
    for i in range(len(levels)):
        for l in range(int(levels[i]) + 1):
            nb = adj[i][l]
            if len(nb) > 1:
                g.append(np.diff(np.sort(o2n[nb])))
    g = np.concatenate(g) if g else np.array([])
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--query", default=None)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--ef", type=int, default=200)
    args = ap.parse_args()

    print("[1/5] loading SIFT base")
    X = read_fvecs(args.base)
    N, D = X.shape
    print(f"      N={N} D={D}")

    print("[2/5] building HNSW (faiss)")
    t0 = time.time()
    idx = build_hnsw(X, args.m, args.ef)
    print(f"      build: {time.time()-t0:.1f}s")

    print("[3/5] extracting adjacency")
    levels, adj = extract_graph(idx.hnsw)
    edges = count_edges(levels, adj)
    orig = original_bytes(levels, adj)
    print(f"      edges={edges} avg_out_deg={edges/N:.2f}  original={orig/1e6:.1f} MB")

    print("[4/5] reordering (RCM + Morton)")
    t0 = time.time()
    A = symmetrized_adjacency(levels, adj, N)
    perm_rcm, o2n_rcm = rcm_order(A)
    print(f"      RCM: {time.time()-t0:.1f}s")
    o2n_morton = morton_order(X)

    print("[5/5] delta encoding")
    for name, o2n in (("RCM", o2n_rcm), ("Morton", o2n_morton)):
        c_varint = delta_encode(levels, adj, o2n, scheme="varint")
        c_fixed8 = delta_encode(levels, adj, o2n, scheme="fixed8")
        g = gap_hist(levels, adj, o2n)
        p1 = np.mean(g < 256) * 100 if len(g) else 0
        p2 = np.mean(g < 65536) * 100 if len(g) else 0
        print(f"      {name:8s}: varint {c_varint/1e6:6.1f} MB ({orig/c_varint:5.2f}x)  "
              f"fixed8 {c_fixed8/1e6:6.1f} MB ({orig/c_fixed8:5.2f}x)  "
              f"gap<1B:{p1:.0f}% <2B:{p2:.0f}%")


if __name__ == "__main__":
    main()
