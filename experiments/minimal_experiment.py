"""Minimal experiment: does graph reordering + delta encoding shrink the HNSW
adjacency lists (B3 vs B0) without changing the topology?

Run:
    .venv/bin/python experiments/minimal_experiment.py --n 100000 --d 128 --m 16
"""

import argparse
import time

import numpy as np

from hnsw_graph import build_hnsw, extract_graph, count_edges
from reorder import (
    symmetrized_adjacency,
    rcm_order,
    delta_encode,
    original_bytes,
    roundtrip_matches,
)


def gap_histogram(levels, adj, old_to_new):
    """Distribution of gap sizes after reordering + sorting (drives the bits needed)."""
    gaps = []
    for i in range(len(levels)):
        for l in range(int(levels[i]) + 1):
            nbs = adj[i][l]
            if len(nbs) <= 1:
                continue
            new_ids = np.sort(old_to_new[nbs])
            gaps.append(np.diff(new_ids))
    if not gaps:
        return np.array([])
    return np.concatenate(gaps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--ef", type=int, default=200)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    data = rng.normal(size=(args.n, args.d)).astype("float32")
    data /= np.linalg.norm(data, axis=1, keepdims=True)  # unit vectors, like RaBitQ's setting

    print(f"[1/6] building HNSW  N={args.n} d={args.d} M={args.m} ef={args.ef}")
    t0 = time.time()
    idx = build_hnsw(data, args.m, args.ef)
    print(f"      build time: {time.time() - t0:.1f}s")

    print("[2/6] extracting adjacency lists")
    levels, adj = extract_graph(idx.hnsw)
    edges = count_edges(levels, adj)
    print(f"      nodes={len(levels)} directed_edges={edges} avg_out_deg={edges/len(levels):.2f}")

    print("[3/6] reordering (RCM)")
    t0 = time.time()
    A = symmetrized_adjacency(levels, adj, len(levels))
    perm, old_to_new = rcm_order(A)
    print(f"      reorder time: {time.time() - t0:.1f}s")

    print("[4/6] delta encoding")
    orig = original_bytes(levels, adj)
    t0 = time.time()
    varint_cost = delta_encode(levels, adj, old_to_new, scheme="varint")
    varint_time = time.time() - t0
    fixed8_cost = delta_encode(levels, adj, old_to_new, scheme="fixed8")
    print(f"      encode time (varint, full pass): {varint_time:.1f}s")

    print("[5/6] verification (roundtrip lossless?)")
    ok = roundtrip_matches(levels, adj, perm, old_to_new)
    print(f"      roundtrip matches: {ok}")

    print("[6/6] gap histogram (post-reorder)")
    gaps = gap_histogram(levels, adj, old_to_new)
    for byte_width in (1, 2, 4):
        frac = np.mean(gaps < (1 << (8 * byte_width)))
        print(f"      gaps fitting in {byte_width}B: {frac*100:.1f}%")

    print("\n===== RESULTS (graph structure only) =====")
    print(f"original  (4B/edge) : {orig/1e6:8.1f} MB")
    print(f"delta varint        : {varint_cost/1e6:8.1f} MB  ({orig/varint_cost:5.2f}x smaller)")
    print(f"delta fixed8        : {fixed8_cost/1e6:8.1f} MB  ({orig/fixed8_cost:5.2f}x smaller)")


if __name__ == "__main__":
    main()
