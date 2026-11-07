"""Compare graph adjacency encodings on SIFT1M (goal: reach ~2x compression)."""

import argparse
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

from io_data import read_fvecs
from hnsw_graph import build_hnsw, extract_graph, count_edges
from reorder import varint_bytes


def gap_bits_and_overflow(sorted_ids, bits_gap):
    """Cost (in bits) of a fixed-bit-width gap encoding + overflow count."""
    if len(sorted_ids) <= 1:
        return 0, 0
    gaps = np.diff(sorted_ids)
    n_overflow = int(np.sum(gaps >= (1 << bits_gap)))
    cost = len(gaps) * bits_gap + n_overflow * 32
    return cost, n_overflow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--n", type=int, default=1_000_000)
    args = ap.parse_args()

    X = read_fvecs(args.base)[:args.n]
    N, D = X.shape
    idx = build_hnsw(X, 16, 200)
    levels, adj = extract_graph(idx.hnsw)
    edges = count_edges(levels, adj)
    orig = edges * 4

    # RCM
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

    # collect sorted neighbor new-ids per (node, layer)
    lists = []
    for i in range(N):
        for l in range(int(levels[i]) + 1):
            nb = adj[i][l]
            if len(nb):
                lists.append(np.sort(o2n[nb]))

    # varint (baseline)
    cost_varint = 0
    for s in lists:
        cost_varint += varint_bytes(int(s[0]))
        for x in np.diff(s):
            cost_varint += varint_bytes(int(x))

    print(f"N={N} edges={edges} orig={orig/1e6:.1f}MB")
    print(f"varint        : {cost_varint/1e6:6.1f}MB  ({orig/cost_varint:5.2f}x)")

    # fixed-bit-width gap encodings, first neighbor stored in ceil(log2 N) bits
    bits_first = int(np.ceil(np.log2(N)))
    for bits_gap in (12, 13, 14, 15, 16, 18):
        cost = 0
        n_ov = 0
        for s in lists:
            cost += bits_first
            c, o = gap_bits_and_overflow(s, bits_gap)
            cost += c
            n_ov += o
        cost_bytes = cost / 8
        ov_frac = n_ov / edges * 100
        print(f"bitpack b={bits_gap:2d}: {cost_bytes/1e6:6.1f}MB  ({orig/cost_bytes:5.2f}x)  overflow={ov_frac:.1f}%")


if __name__ == "__main__":
    main()
