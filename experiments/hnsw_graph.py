"""Build an HNSW index (via faiss) and extract its adjacency lists.

Why faiss instead of hnswlib: hnswlib's Python bindings do not expose the
per-node neighbor lists, while faiss's `IndexHNSW` exposes them directly as
`levels` / `neighbors` / `offsets` / `cum_nneighbor_per_level`, which is exactly
what graph reordering + delta encoding needs.
"""

import numpy as np
import faiss


def build_hnsw(data: np.ndarray, M: int, ef_construction: int) -> faiss.IndexHNSWFlat:
    """Build an HNSW index over `data` (float32, shape [N, D]).

    Note on faiss semantics: `M` is the number of neighbors kept per node in the
    upper layers, and layer 0 keeps `2*M`. This matches the HNSW paper's
    "Mmax = M, Mmax0 = 2*M" convention.
    """
    N, D = data.shape
    data = np.ascontiguousarray(data, dtype=np.float32)
    idx = faiss.IndexHNSWFlat(D, M)
    idx.hnsw.efConstruction = ef_construction
    idx.add(data)
    return idx


def extract_graph(hnsw) -> tuple[np.ndarray, list]:
    """Extract the HNSW adjacency lists.

    Returns
    -------
    levels : np.ndarray (int32, shape [N])
        The maximum layer of each node.
    adj : list of list of np.ndarray
        adj[i][l] is the (directed) neighbor ids of node `i` at layer `l`
        (original ids, `-1` padding slots removed, self-loops removed).
    """
    levels = faiss.vector_to_array(hnsw.levels).astype(np.int32)
    neighbors = faiss.vector_to_array(hnsw.neighbors).astype(np.int64)
    offsets = faiss.vector_to_array(hnsw.offsets).astype(np.int64)
    cum = faiss.vector_to_array(hnsw.cum_nneighbor_per_level).astype(np.int64)

    N = len(levels)
    adj = [None] * N
    for i in range(N):
        li = int(levels[i])
        node_layers = []
        for l in range(li + 1):
            s = offsets[i] + cum[l]
            e = offsets[i] + cum[l + 1]
            nb = neighbors[s:e]
            nb = nb[nb >= 0]          # drop unused (-1) slots
            nb = nb[nb != i]          # drop self-loops
            node_layers.append(nb)
        adj[i] = node_layers
    return levels, adj


def count_edges(levels, adj) -> int:
    """Total number of directed edges (== stored link slots used)."""
    total = 0
    for i in range(len(levels)):
        for l in range(int(levels[i]) + 1):
            total += len(adj[i][l])
    return total
