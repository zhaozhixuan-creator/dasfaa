"""Loaders for the IRISA texmex .fvecs / .ivecs formats.

Each record is: int32 dim followed by dim values (float32 for fvecs, int32
for ivecs).
"""

import numpy as np


def read_fvecs(path):
    data = np.fromfile(path, dtype=np.float32)
    dim = int(data.view(np.int32)[0])
    assert dim > 0, f"bad dim {dim} in {path}"
    data = data.reshape(-1, 1 + dim)[:, 1:].copy()  # drop the per-vector dim prefix
    return np.ascontiguousarray(data)


def read_ivecs(path):
    data = np.fromfile(path, dtype=np.int32)
    dim = int(data[0])
    assert dim > 0, f"bad dim {dim} in {path}"
    data = data.reshape(-1, 1 + dim)[:, 1:].copy()
    return np.ascontiguousarray(data)
