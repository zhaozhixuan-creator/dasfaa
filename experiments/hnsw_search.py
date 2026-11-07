"""Correct HNSW search (hierarchical greedy + layer-0 beam), pluggable distance.

Implements the HNSW search procedure (paper Algorithms 2 & 5) over the graph
extracted from faiss, so we can swap exact vs RaBitQ-estimated distances.
"""

import heapq


def search_layer(adj_layers, lc, dist, ep, ef):
    """Beam search on layer `lc`, returning up to `ef` nearest node ids."""
    visited = {ep}
    cand = [(dist(ep), ep)]          # min-heap (distance, id)
    W = [(-dist(ep), ep)]            # max-heap of the ef nearest (neg dist)
    while cand:
        cdist, c = heapq.heappop(cand)
        if cdist > -W[0][0]:         # nearest unvisited candidate is farther than worst kept
            break
        for v in adj_layers[c][lc]:
            if v in visited:
                continue
            visited.add(v)
            d = dist(v)
            if d < -W[0][0] or len(W) < ef:
                heapq.heappush(cand, (d, v))
                heapq.heappush(W, (-d, v))
                if len(W) > ef:
                    heapq.heappop(W)
    return [x[1] for x in W]


def hnsw_search(levels, adj_layers, entry, dist, ef, K):
    """Full HNSW search: greedy descent from top layer, then beam on layer 0."""
    L = int(levels[entry])
    ep = entry
    for lc in range(L, 0, -1):
        # simple greedy (ef=1): always move to the closest neighbor at this layer
        improved = True
        while improved:
            improved = False
            if lc >= len(adj_layers[ep]):   # current node has no neighbors at this layer
                break
            best = ep
            best_d = dist(ep)
            for v in adj_layers[ep][lc]:
                d = dist(v)
                if d < best_d:
                    best_d = d
                    best = v
            if best != ep:
                ep = best
                improved = True
    W = search_layer(adj_layers, 0, dist, ep, ef)
    W.sort(key=lambda v: dist(v))
    return W[:K]
