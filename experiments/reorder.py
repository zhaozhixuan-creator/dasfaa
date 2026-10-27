"""Graph reordering (RCM) and delta encoding for the HNSW adjacency lists."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee


def symmetrized_adjacency(levels, adj, N) -> csr_matrix:
    """Undirected (union) adjacency matrix of the whole multi-layer graph.

    Reordering needs an undirected connectivity structure, so a directed edge
    u->v contributes an undirected edge {u, v}.
    """
    rows = []
    cols = []
    for i in range(N):
        for l in range(int(levels[i]) + 1):
            nbs = adj[i][l]
            if len(nbs):
                rows.append(np.full(len(nbs), i, dtype=np.int32))
                cols.append(nbs.astype(np.int32))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    data = np.ones(len(rows), dtype=np.int8)
    A = csr_matrix((data, (rows, cols)), shape=(N, N))
    A = (A + A.T).tocsr()
    A.data[:] = 1
    return A


def rcm_order(A: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Compute a Reverse Cuthill-McKee ordering.

    Returns
    -------
    perm : np.ndarray
        perm[new_id] = old_id  (new_id k holds the node that was old_id perm[k]).
    old_to_new : np.ndarray
        old_to_new[old_id] = new_id.
    """
    perm = reverse_cuthill_mckee(A, symmetric_mode=True).astype(np.int64)
    old_to_new = np.empty(len(perm), dtype=np.int64)
    old_to_new[perm] = np.arange(len(perm), dtype=np.int64)
    return perm, old_to_new


def varint_bytes(x: int) -> int:
    """LEB128-style byte count for a non-negative integer."""
    n = 1
    while x >= 128:
        x >>= 7
        n += 1
    return n


def _encode_node(neighbors_new_sorted: np.ndarray, scheme: str) -> int:
    """Bytes to encode one node's sorted neighbor list.

    `scheme` is one of:
      - "varint": first id + consecutive gaps, LEB128.
      - "fixed8": first id (4B) + gaps (1B each); a gap > 255 falls back to 4B.
    """
    k = len(neighbors_new_sorted)
    if k == 0:
        return 0
    first = int(neighbors_new_sorted[0])
    gaps = np.diff(neighbors_new_sorted)
    if scheme == "varint":
        cost = varint_bytes(first)
        for g in gaps:
            cost += varint_bytes(int(g))
        return cost
    if scheme == "fixed8":
        cost = 4  # first id stored as 4 bytes
        for g in gaps:
            g = int(g)
            cost += 1 if g <= 255 else 4  # overflow -> full 4 bytes
        return cost
    raise ValueError(f"unknown scheme: {scheme}")


def delta_encode(levels, adj, old_to_new, scheme: str = "varint") -> int:
    """Total bytes of the reordered + delta-encoded graph (all layers).

    A single global renumbering (from `old_to_new`) is applied to every layer;
    each node's neighbor list is sorted by new id and encoded as first + gaps.
    """
    N = len(levels)
    total = 0
    for i in range(N):
        for l in range(int(levels[i]) + 1):
            nbs = adj[i][l]
            if len(nbs) == 0:
                continue
            new_ids = np.sort(old_to_new[nbs])
            total += _encode_node(new_ids, scheme)
    return total


def original_bytes(levels, adj) -> int:
    """Original storage: each directed link as a 4-byte int (Malkov's model)."""
    total = 0
    for i in range(len(levels)):
        for l in range(int(levels[i]) + 1):
            total += len(adj[i][l])
    return total * 4


def roundtrip_matches(levels, adj, perm, old_to_new, sample=2000) -> bool:
    """Verify the renumber + sort + delta chain is lossless.

    For a random subset of nodes, confirm that mapping neighbors to new ids,
    sorting, encoding (first + gaps), decoding, and mapping back reproduces the
    exact original neighbor *set* (topology preservation -> recall preserved).
    """
    N = len(levels)
    rng = np.random.default_rng(0)
    nodes = rng.choice(N, size=min(sample, N), replace=False)
    for i in nodes:
        for l in range(int(levels[i]) + 1):
            orig = set(int(x) for x in adj[i][l])
            if not orig:
                continue
            sorted_new = np.sort(old_to_new[list(orig)])
            # encode -> decode roundtrip for varint-style first + gaps
            first = sorted_new[0]
            gaps = np.diff(sorted_new)
            decoded_new = np.concatenate(([first], first + np.cumsum(gaps)))
            back = set(int(x) for x in perm[decoded_new])
            if back != orig:
                return False
    return True
