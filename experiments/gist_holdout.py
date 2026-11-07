import numpy as np, faiss, time
from io_data import read_fvecs
from hnsw_graph import build_hnsw, extract_graph
from hnsw_search import hnsw_search
from rabitq_impl import rabitq_index, estimate_distances

X = read_fvecs('../data/gist/gist_base.fvecs')[:1000000]
N, D = X.shape
# holdout: last 1000 as queries, first 999000 as base
NBASE = 999000
base = X[:NBASE]
Q = X[NBASE:]
rng = np.random.default_rng(0)
qidx = rng.choice(len(Q), size=100, replace=False)
Q = Q[qidx]

bf = faiss.IndexFlatL2(D); bf.add(base)
_, GT = bf.search(Q, 10)

idx = build_hnsw(base, 16, 200)
levels, adj = extract_graph(idx.hnsw)
entry = int(idx.hnsw.entry_point)
codes, P, x0, c = rabitq_index(base)

idx.hnsw.efSearch = 128
_, Ih = idx.search(Q, 10)
r0 = np.mean([len(set(Ih[i]) & set(GT[i]))/10 for i in range(100)])
print(f'GIST holdout: exact recall@10 = {r0:.3f}')

for ef in (512, 2000):
    hits=0; t=time.time()
    for qi in range(100):
        q=Q[qi]
        est_d,_=estimate_distances(codes,P,x0,c,q)
        df=lambda v: float(est_d[v])
        res=hnsw_search(levels, adj, entry, df, ef, 2000)
        exact_d=np.sum((base[res]-q)**2,axis=1)
        res=list(np.array(res)[np.argsort(exact_d)][:10])
        hits+=len(set(res)&set(GT[qi]))
    print(f'GIST holdout: rabitq ef={ef} recall@10 = {hits/1000:.3f} [{time.time()-t:.0f}s]')
