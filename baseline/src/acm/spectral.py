from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla


def laplacian(adj: np.ndarray) -> np.ndarray:
    deg = adj.sum(axis=1).astype(np.float64)
    return np.diag(deg) - adj.astype(np.float64)


def algebraic_connectivity(adj: np.ndarray) -> float:
    vals = np.linalg.eigvalsh(laplacian(adj))
    if len(vals) < 2:
        return 0.0
    return float(max(vals[1], 0.0))


def _normalize_eigvec(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float64, copy=True)
    v -= v.mean()
    nrm = np.linalg.norm(v)
    if nrm < 1e-12:
        return np.zeros_like(v)
    return v / nrm


def _project_and_orthonormalize_subspace(V: np.ndarray) -> np.ndarray:
    """
    Project columns onto 1^perp and return an orthonormal basis (QR).
    """
    if V.ndim != 2:
        raise ValueError(f"V must be 2D, got shape={V.shape}")
    n = int(V.shape[0])
    if n == 0 or int(V.shape[1]) == 0:
        return np.zeros((n, 0), dtype=np.float64)
    W = np.asarray(V, dtype=np.float64, copy=True)
    one = np.ones((n, 1), dtype=np.float64)
    W = W - one * ((one.T @ W) / float(n))
    Q, R = np.linalg.qr(W, mode="reduced")
    if R.size == 0:
        return np.zeros((n, 0), dtype=np.float64)
    diag = np.abs(np.diag(R))
    keep = diag > 1e-10
    if not np.any(keep):
        return np.zeros((n, 0), dtype=np.float64)
    return np.asarray(Q[:, keep], dtype=np.float64)


def _cluster_tol(lam2: float, eps_abs: float, eps_rel: float) -> float:
    return float(max(float(eps_abs), float(eps_rel) * max(1.0, abs(float(lam2)))))


def lambda2_eigenspace_basis(
    adj: np.ndarray,
    *,
    eps_abs: float = 1e-12,
    eps_rel: float = 1e-10,
    eigsh_cutoff_n: int = 96,
    sparse_k_init: int = 6,
    sparse_k_step: int = 4,
    sparse_k_max: int = 96,
) -> Tuple[float, np.ndarray]:
    """
    Return (lambda2, V) where columns of V form an orthonormal basis of the
    lambda2 eigenspace (within tolerance).

    For n <= eigsh_cutoff_n, uses dense eigh.
    For larger n, adaptively expands eigsh(k) until the lambda2 cluster
    is not truncated, then extracts/orthonormalizes the cluster basis.
    Falls back to dense eigh on sparse failures.
    """
    n = int(adj.shape[0])
    if n < 2:
        return 0.0, np.zeros((n, 0), dtype=np.float64)

    L = laplacian(adj)

    def _extract(vals: np.ndarray, vecs: np.ndarray) -> Tuple[float, np.ndarray]:
        order = np.argsort(vals)
        vals = np.asarray(vals[order], dtype=np.float64)
        vecs = np.asarray(vecs[:, order], dtype=np.float64)
        lam2 = float(max(vals[1], 0.0))
        tol = _cluster_tol(lam2, eps_abs=float(eps_abs), eps_rel=float(eps_rel))
        idx = np.flatnonzero(np.abs(vals - lam2) <= tol)
        idx = idx[idx >= 1]
        if idx.size == 0:
            idx = np.asarray([1], dtype=np.int64)
        V = _project_and_orthonormalize_subspace(vecs[:, idx])
        return lam2, V

    if n <= int(eigsh_cutoff_n):
        vals, vecs = np.linalg.eigh(L)
        return _extract(vals, vecs)

    Ls = sparse.csr_matrix(L)
    k = max(3, int(sparse_k_init))
    k_cap = max(k, min(n - 1, int(sparse_k_max)))
    try:
        while True:
            k = min(k, n - 1)
            vals, vecs = spla.eigsh(Ls, k=k, which="SM", tol=1e-6, maxiter=5000)
            order = np.argsort(vals)
            vals_s = np.asarray(vals[order], dtype=np.float64)
            vecs_s = np.asarray(vecs[:, order], dtype=np.float64)
            lam2 = float(max(vals_s[1], 0.0))
            tol = _cluster_tol(lam2, eps_abs=float(eps_abs), eps_rel=float(eps_rel))
            idx = np.flatnonzero((np.abs(vals_s - lam2) <= tol) & (np.arange(vals_s.size) >= 1))
            if idx.size == 0:
                idx = np.asarray([1], dtype=np.int64)
            # If the cluster touches the highest computed eigenvalue, request more.
            touches_top = int(idx.max()) >= (vals_s.size - 1)
            if touches_top and k < min(n - 1, k_cap):
                k = min(n - 1, k + int(max(1, sparse_k_step)))
                continue
            V = _project_and_orthonormalize_subspace(vecs_s[:, idx])
            return lam2, V
    except Exception:
        vals, vecs = np.linalg.eigh(L)
        return _extract(vals, vecs)


def lambda2_subspace_edge_scores(
    adj: np.ndarray,
    candidate_uv: Sequence[Sequence[int]] | np.ndarray,
    *,
    eps_abs: float = 1e-12,
    eps_rel: float = 1e-10,
    eigsh_cutoff_n: int = 96,
    sparse_k_init: int = 6,
    sparse_k_step: int = 4,
    sparse_k_max: int = 96,
) -> Tuple[float, np.ndarray]:
    """
    Compute multiplicity-aware lambda2-eigenspace edge scores:
      score(u,v) = sum_k (V[u,k] - V[v,k])^2
    where columns of V span the lambda2 eigenspace.
    """
    uv = np.asarray(candidate_uv, dtype=np.int64)
    if uv.size == 0:
        lam2 = algebraic_connectivity(adj)
        return float(lam2), np.zeros(0, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"candidate_uv must have shape (p,2), got {uv.shape}")

    lam2, V = lambda2_eigenspace_basis(
        adj,
        eps_abs=float(eps_abs),
        eps_rel=float(eps_rel),
        eigsh_cutoff_n=int(eigsh_cutoff_n),
        sparse_k_init=int(sparse_k_init),
        sparse_k_step=int(sparse_k_step),
        sparse_k_max=int(sparse_k_max),
    )
    if V.shape[1] == 0:
        _, phi2, _ = two_smallest_nontrivial(adj, eigsh_cutoff_n=int(eigsh_cutoff_n))
        scores = (phi2[uv[:, 0]] - phi2[uv[:, 1]]) ** 2
        return float(lam2), np.asarray(scores, dtype=np.float64)

    diff = V[uv[:, 0], :] - V[uv[:, 1], :]
    scores = np.einsum("ij,ij->i", diff, diff)
    return float(lam2), np.asarray(scores, dtype=np.float64)


def two_smallest_nontrivial(
    adj: np.ndarray,
    warm_start: Optional[np.ndarray] = None,
    eigsh_cutoff_n: int = 96,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Returns (lambda2, phi2, phi3) with phi vectors normalized.
    Falls back to dense eigendecomposition for small graphs.
    """
    n = adj.shape[0]
    L = laplacian(adj)
    if n <= eigsh_cutoff_n:
        vals, vecs = np.linalg.eigh(L)
        lam2 = float(max(vals[1], 0.0)) if n >= 2 else 0.0
        phi2 = _normalize_eigvec(vecs[:, 1]) if n >= 2 else np.zeros(n)
        phi3 = _normalize_eigvec(vecs[:, 2]) if n >= 3 else np.zeros(n)
        return lam2, phi2, phi3

    # Sparse path for larger n.
    Ls = sparse.csr_matrix(L)
    v0 = warm_start
    try:
        vals, vecs = spla.eigsh(Ls, k=3, which="SM", v0=v0, tol=1e-5, maxiter=3000)
        order = np.argsort(vals)
        vals = vals[order]
        vecs = vecs[:, order]
        lam2 = float(max(vals[1], 0.0))
        phi2 = _normalize_eigvec(vecs[:, 1])
        phi3 = _normalize_eigvec(vecs[:, 2] if vecs.shape[1] > 2 else np.zeros(n))
        return lam2, phi2, phi3
    except Exception:
        vals, vecs = np.linalg.eigh(L)
        lam2 = float(max(vals[1], 0.0)) if n >= 2 else 0.0
        phi2 = _normalize_eigvec(vecs[:, 1]) if n >= 2 else np.zeros(n)
        phi3 = _normalize_eigvec(vecs[:, 2]) if n >= 3 else np.zeros(n)
        return lam2, phi2, phi3


def grounded_laplacian_inverse(adj: np.ndarray, ground: int = 0) -> np.ndarray:
    n = adj.shape[0]
    if not (0 <= ground < n):
        raise ValueError(f"ground out of range: {ground}")
    L = laplacian(adj)
    keep = [i for i in range(n) if i != ground]
    Lg = L[np.ix_(keep, keep)]
    # Connected graph guarantees SPD. Add tiny ridge for numeric stability.
    ridge = 1e-10 * np.eye(Lg.shape[0], dtype=np.float64)
    return np.linalg.inv(Lg + ridge)


def grounded_incidence_vector(u: int, v: int, n: int, ground: int = 0) -> np.ndarray:
    g = np.zeros(n - 1, dtype=np.float64)
    for node, sign in ((u, 1.0), (v, -1.0)):
        if node == ground:
            continue
        idx = node if node < ground else node - 1
        g[idx] += sign
    return g


def effective_resistance_from_grounded_inv(
    inv_lg: np.ndarray,
    u: int,
    v: int,
    n: int,
    ground: int = 0,
) -> float:
    if u == v:
        return 0.0
    if u == ground:
        vi = v if v < ground else v - 1
        return float(max(inv_lg[vi, vi], 0.0))
    if v == ground:
        ui = u if u < ground else u - 1
        return float(max(inv_lg[ui, ui], 0.0))
    ui = u if u < ground else u - 1
    vi = v if v < ground else v - 1
    val = inv_lg[ui, ui] + inv_lg[vi, vi] - 2.0 * inv_lg[ui, vi]
    return float(max(val, 0.0))
