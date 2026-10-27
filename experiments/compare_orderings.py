"""Compare graph orderings for delta compression of the HNSW adjacency lists.

Orderings: identity (no reorder), RCM (bandwidth/BFS), spectral (Fiedler vector),
recursive spectral bisection. Reports varint-compressed size and gap histogram.
"""

import argparse
import time

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import eigsh

from io_data import read_fvecs
from hnsw_graph import build_hnsw, extract_graph, count_edges
from reorder import symmetrized_adjacency, delta_encode, original_bytes, varint_bytes


def spectral_order(A, min_size=256):
    """Recursive spectral bisection: order leaves by Fiedler vector recursively."""
    N = A.shape[0]
    order = np.empty(N, dtype=np.int64)

    def recurse(nodes, lo):
        m = len(nodes)
        if m <= min_size:
            order[lo:lo + m] = nodes
            return
        sub = A[nodes][:, nodes]
        d = np.asarray(sub.sum(axis=1)).ravel().astype(np.float64)
        d_inv_sqrt = np.zeros_like(d)
        d_inv_sqrt[d > 0] = 1.0 / np.sqrt(d[d > 0])
        An = sub.multiply(d_inv_sqrt[:, None]).multiply(d_inv_sqrt[None, :]).tocsr()
        _, vecs = eigsh(An, k=2, which="LA")
        f = vecs[:, 1]
        split = m // 2
        idx = np.argpartition(f, split)
        left = nodes[idx[:split]]
        right = nodes[idx[split:]]
        recurse(left, lo)
        recurse(right, lo + split)

    recurse(np.arange(N), 0)
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--ef", type=int, default=200)
    args = ap.parse_args()

    X = read_fvecs(args.base)[:args.n]
    N, D = X.shape
    print(f"N={N} D={D}")

    idx = build_hnsw(X, args.m, args.ef)
    levels, adj = extract_graph(idx.hnsw)
    edges = count_edges(levels, adj)
    orig = original_bytes(levels, adj)
    print(f"edges={edges} avg_deg={edges/N:.2f} original={orig/1e6:.1f}MB")

    A = symmetrized_adjacency(levels, adj, N)

    # orderings
    orderings = {}
    orderings["identity"] = np.arange(N, dtype=np.int64)

    t = time.time()
    perm = reverse_cuthill_mckee(A, symmetric_mode=True).astype(np.int64)
    o2n = np.empty(N, dtype=np.int64); o2n[perm] = np.arange(N)
    orderings["RCM"] = o2n
    print(f"RCM: {time.time()-t:.1f}s")

    t = time.time()
    orderings["spectral"] = spectral_order(A)
    print(f"spectral (recursive): {time.time()-t:.1f}s")

    for name, o2n in orderings.items():
        c = delta_encode(levels, adj, o2n, scheme="varint")
        # gap histogram
        gaps = []
        for i in range(N):
            for l in range(int(levels[i]) + 1):
                nb = adj[i][l]
                if len(nb) > 1:
                    gaps.append(np.diff(np.sort(o2n[nb])))
        gaps = np.concatenate(gaps)
        p1 = np.mean(gaps < 256) * 100
        p2 = np.mean(gaps < 65536) * 100
        p4 = np.mean(gaps < 64) * 100
        print(f"  {name:10s}: varint {c/1e6:6.1f}MB ({orig/c:5.2f}x)  gap<64:{p4:.0f}% <256:{p1:.0f}% <65536:{p2:.0f}%")


if __name__ == "__main__":
    main()
