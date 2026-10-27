# SIFT1M Dataset

This directory holds the SIFT1M benchmark dataset used by the experiments in
[`experiments/`](../experiments/). The data files themselves are **not** tracked
in git (~700 MB uncompressed), so download and extract them here before running
any experiment.

## Download

SIFT1M is distributed as part of the IRISA *TexMex* corpus:

```bash
cd data
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
```

After extraction the layout should be:

```
data/
├── README.md
└── sift/
    ├── sift_base.fvecs          # 1,000,000 base vectors (128-d float32), ~500 MB
    ├── sift_query.fvecs         # 10,000 query vectors (128-d float32), ~5 MB
    ├── sift_learn.fvecs         # 100,000 learning vectors (128-d float32), ~50 MB
    └── sift_groundtruth.ivecs   # 100 nearest neighbors per query (int32 ids), ~4 MB
```

The experiments load the data from `../data/sift/sift_base.fvecs`,
`../data/sift/sift_query.fvecs` and `../data/sift/sift_groundtruth.ivecs` — see
[`experiments/sift_experiment.py`](../experiments/sift_experiment.py).

## File format

- `.fvecs` — sequence of records, each `[int32 dim][float32 × dim]`.
- `.ivecs` — sequence of records, each `[int32 dim][int32 × dim]`.

The leading `int32` of every record is the vector dimension, followed by the
vector entries. See `read_fvecs` / `read_ivecs` in
[`experiments/io_data.py`](../experiments/io_data.py).
