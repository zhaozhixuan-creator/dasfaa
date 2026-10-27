"""Estimate several encodings of the reordered HNSW adjacency lists on SIFT.

Encodings:
  - varint gaps (baseline, already measured)
  - signed delta from the node's own id (zigzag varint)
  - 2-level block encoding: distinct blocks (varint block ids) + in-block offsets
Also reports the distinct-block count, which determines whether 2-level wins.
"""

import argparse
import time

import numpy as np

from io_data import read_fvecs
from hnsw_graph import build_hnsw, extract_graph, count_edges
from reorder import symmetrized_adjacency, rcm_order, varint_bytes


def zigzag(x):
    return 2 * x if x >= 0 else -2 * x - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--ef", type=int, default=200)
    args = ap.parse_args()

    X = read_fvecs(args.base)[:args.n]
    N, D = X.shape
    idx = build_hnsw(X, args.m, args.ef)
    levels, adj = extract_graph(idx.hnsw)
    edges = count_edges(levels, adj)
    orig = edges * 4
    A = symmetrized_adjacency(levels, adj, N)
    _, o2n = rcm_order(A)
    print(f"N={N} edges={edges} avg_deg={edges/N:.2f} original={orig/1e6:.1f}MB")

    # collect all neighbor lists (new ids), as lists
    all_nb = []
    for i in range(N):
        for l in range(int(levels[i]) + 1):
            nb = adj[i][l]
            if len(nb):
                all_nb.append(np.sort(o2n[nb]))

    # 1) varint gaps
    cost_gap = 0
    for nb in all_nb:
        cost_gap += varint_bytes(int(nb[0]))
        for x in np.diff(nb):
            cost_gap += varint_bytes(int(x))

    # 2) signed delta from node (need node's own new id)
    cost_signed = 0
    for i in range(N):
        vi = int(o2n[i])
        for l in range(int(levels[i]) + 1):
            nb = adj[i][l]
            if not len(nb):
                continue
            for x in np.sort(o2n[nb]):
                cost_signed += varint_bytes(zigzag(int(x) - vi))

    # 3) 2-level block encoding
    for S in (32, 64, 128, 256):
        nblocks = (N + S - 1) // S
        block_id_bits = int(np.ceil(np.log2(nblocks))) if nblocks > 1 else 1
        offset_bits = int(np.ceil(np.log2(S)))
        cost = 0
        ndist_total = 0
        for nb in all_nb:
            blocks = np.unique(nb // S)
            ndist_total += len(blocks)
            # store distinct block ids (varint) + one offset per neighbor (fixed offset_bits)
            for b in blocks:
                cost += varint_bytes(int(b))
            cost += len(nb) * offset_bits
        cost_bytes = cost / 8
        avg_distinct = ndist_total / len(all_nb)
        print(f"  block S={S}: {cost_bytes/1e6:6.1f}MB ({orig/cost_bytes:5.2f}x)  avg_distinct_blocks={avg_distinct:.1f}")

    print(f"  varint gaps : {cost_gap/1e6:6.1f}MB ({orig/cost_gap:5.2f}x)")
    print(f"  signed delta: {cost_signed/1e6:6.1f}MB ({orig/cost_signed:5.2f}x)")


if __name__ == "__main__":
    main()
