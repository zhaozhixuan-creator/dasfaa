"""GIST1M (960-dim) with M=32 (appropriate for high-dim), holdout query.

Recomputes B0/B2/B3 memory (Golomb-Rice graph compression) and recall.
"""

import numpy as np
import faiss
import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

from io_data import read_fvecs
from hnsw_graph import build_hnsw, extract_graph, count_edges
from hnsw_search import hnsw_search
from rabitq_impl import rabitq_index, estimate_distances


def golomb_rice_bytes(levels, adj, o2n, k=15):
    bits_first = int(np.ceil(np.log2(len(levels))))
    total = 0
    for i in range(len(levels)):
        for l in range(int(levels[i]) + 1):
            nb = adj[i][l]
            if len(nb) == 0:
                continue
            s = np.sort(o2n[nb])
            total += bits_first
            g = np.diff(s)
            total += (g >> k).sum() + len(g) * (1 + k)
    return total / 8


def main():
    M = 32
    X = read_fvecs('../data/gist/gist_base.fvecs')[:1000000]
    N, D = X.shape
    NBASE = 999800
    base = X[:NBASE]
    Q = X[NBASE:]  # 200 holdout queries
    nq = len(Q)

    bf = faiss.IndexFlatL2(D); bf.add(base)
    _, GT = bf.search(Q, 10)

    print(f"building HNSW M={M} on GIST (N={NBASE}, D={D})")
    idx = build_hnsw(base, M, 200)
    levels, adj = extract_graph(idx.hnsw)
    edges = count_edges(levels, adj)
    graph_raw = edges * 4

    rows, cols = [], []
    for i in range(NBASE):
        for l in range(int(levels[i]) + 1):
            nb = adj[i][l]
            if len(nb):
                rows.append(np.full(len(nb), i, np.int32))
                cols.append(nb.astype(np.int32))
    rows = np.concatenate(rows); cols = np.concatenate(cols)
    A = csr_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(NBASE, NBASE))
    A = (A + A.T).tocsr(); A.data[:] = 1
    perm = reverse_cuthill_mckee(A, symmetric_mode=True).astype(np.int64)
    o2n = np.empty(NBASE, dtype=np.int64); o2n[perm] = np.arange(NBASE)
    graph_comp = golomb_rice_bytes(levels, adj, o2n)

    print("quantizing RaBitQ (960-dim)")
    codes, P, x0, c = rabitq_index(base)

    vec_raw = NBASE * D * 4
    vec_code = NBASE * (D // 8)
    print("\n===== GIST M=%d MEMORY =====" % M)
    print(f"  B0 HNSW          : {(vec_raw + graph_raw)/1e6:7.1f} MB")
    print(f"  B2 HNSW+RaBitQ   : {(vec_code + graph_raw)/1e6:7.1f} MB")
    print(f"  B3 ours          : {(vec_code + graph_comp)/1e6:7.1f} MB")
    print(f"  graph compression: {graph_raw/1e6:.1f}MB -> {graph_comp/1e6:.1f}MB ({graph_raw/graph_comp:.2f}x)")
    print(f"  total B0->B3     : {(vec_raw+graph_raw)/(vec_code+graph_comp):.1f}x")

    print("\n===== GIST M=%d RECALL@10 =====" % M)
    idx.hnsw.efSearch = 128
    _, Ih = idx.search(Q, 10)
    r0 = np.mean([len(set(Ih[i]) & set(GT[i])) / 10 for i in range(nq)])
    print(f"  exact              : {r0:.3f}")

    entry = int(idx.hnsw.entry_point)
    hits = 0
    t = time.time()
    for qi in range(nq):
        q = Q[qi]
        est_d, _ = estimate_distances(codes, P, x0, c, q)
        df = lambda v: float(est_d[v])
        res = hnsw_search(levels, adj, entry, df, 2000, 2000)
        exact_d = np.sum((base[res] - q) ** 2, axis=1)
        res = list(np.array(res)[np.argsort(exact_d)][:10])
        hits += len(set(res) & set(GT[qi]))
    r2 = hits / (nq * 10)
    print(f"  RaBitQ ef=2000     : {r2:.3f}  [{time.time()-t:.0f}s]")


if __name__ == "__main__":
    main()
