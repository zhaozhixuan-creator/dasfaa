"""RaBitQ quantization + distance estimation (numpy reference implementation).

Follows the recipe in the RaBitQ paper (Gao & Long, SIGMOD'24):
  - index: normalize, random orthogonal transform P, D-bit codes, x0 = <o_bar, o>
  - query: q' = P^T q, quantize to 4-bit q_bar; estimate <o,q> = <x_bar_b, q_bar> / x0
"""

import numpy as np


def rabitq_index(X):
    """Quantize X (N, D) float32 -> (codes N×(D/8) uint8, P (D,D), x0 (N,), c (D,))."""
    N, D = X.shape
    X = X.astype(np.float64)
    c = X.mean(axis=0)
    Xc = X - c
    norms = np.linalg.norm(Xc, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = Xc / norms                      # unit vectors
    G = np.random.default_rng(0).normal(size=(D, D))
    P, _ = np.linalg.qr(G)               # random orthogonal matrix
    XP = Xn @ P                          # inverse transform (P^T = P^-1, so use P)
    # NOTE: Xn @ P = Xn @ P ; the paper computes P^{-1} o. Since P orthogonal,
    # P^{-1} = P^T, and (Xn @ P)[i] = <o_i, P[:, j]> = (P^T o_i)[j]. Correct.
    bin_XP = (XP > 0)                    # D-bit codes (bool)
    codes = np.packbits(bin_XP, axis=1)  # N x (D/8) uint8
    x0 = np.abs(XP).sum(axis=1) / (np.sqrt(D) * np.linalg.norm(XP, axis=1))
    x0[~np.isfinite(x0)] = 0.8
    return codes.astype(np.uint8), P.astype(np.float64), x0.astype(np.float64), c.astype(np.float64)


def _quantize_query(q, P, c, Bq=4):
    """Transform + quantize a raw query to 4-bit q_bar (per-dim unsigned ints)."""
    q = q.astype(np.float64)
    qc = (q - c)
    nq = np.linalg.norm(qc)
    if nq == 0:
        nq = 1.0
    qc = qc / nq
    qp = P.T @ qc                        # P^{-1} q = P^T q
    vmin, vmax = qp.min(), qp.max()
    delta = (vmax - vmin) / (2 ** Bq - 1)
    if delta == 0:
        delta = 1.0
    qbar = np.clip(np.round((qp - vmin) / delta), 0, 2 ** Bq - 1).astype(np.uint8)
    return qbar, delta, vmin, qp


def estimate_distances(codes, P, x0, c, q, Bq=4):
    """Approximate squared distances ||o - q||^2 for all data vectors.

    Returns (dists, <o,q>_est) where dists = ||o_r - q_r||^2 approx.
    """
    N = codes.shape[0]
    D = P.shape[0]
    qbar, delta, vmin, qp = _quantize_query(q, P, c, Bq)
    # acc = <x_bar_b, q_bar_u> via bit-planes of q_bar_u (equation 22 in paper)
    # codes: N x (D/8) uint8, each row is the D-bit binary code
    codes_bits = np.unpackbits(codes, axis=1)[:, :D].astype(np.uint8)  # N x D (0/1)
    acc = np.zeros(N, dtype=np.float64)
    for j in range(Bq):
        qbit = ((qbar >> j) & 1).astype(np.uint8)  # D bits of j-th plane
        acc += (2 ** j) * (codes_bits * qbit[None, :]).sum(axis=1)
    ones_b = codes_bits.sum(axis=1)  # number of 1s in each D-bit code
    # <x_bar, q_bar> = (2Δ·<x̄_b,q̄_u> + 2v_l·Σx̄_b - Δ·Σq̄_u - D·v_l) / √D   (equation 20)
    xq = (2.0 * delta * acc + 2.0 * vmin * ones_b
          - delta * qbar.astype(np.float64).sum() - D * vmin) / np.sqrt(D)
    inner_oq = xq / x0            # unbiased estimator of <o, q>
    # squared distance between unit vectors: ||o - q||^2 = 2 - 2<o,q>
    dist = 2.0 - 2.0 * inner_oq
    return dist, inner_oq
