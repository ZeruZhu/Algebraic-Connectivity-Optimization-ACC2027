from __future__ import annotations
from collections import deque
from typing import List, Tuple

import numpy as np

try:
    from scipy.linalg import eigvalsh as scipy_eigvalsh
except ImportError:  # pragma: no cover - optional dependency
    scipy_eigvalsh = None

try:
    from scipy.sparse.csgraph import shortest_path as scipy_shortest_path
except Exception:  # pragma: no cover - optional dependency
    scipy_shortest_path = None


def build_path_adjacency(n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    return adj


def complete_edge_count(n: int) -> int:
    return (n * (n - 1)) // 2


def normalized_density(n: int, m: int) -> float:
    min_m = n - 1
    max_m = complete_edge_count(n)
    denom = max_m - min_m
    if denom <= 0:
        return 0.0
    return float(m - min_m) / float(denom)


def m_target_from_rho(n: int, rho: float) -> int:
    rho = max(0.0, min(1.0, rho))
    min_m = n - 1
    max_m = complete_edge_count(n)
    m = min_m + rho * (max_m - min_m)
    m_rounded = int(round(m))
    return max(min_m, min(max_m, m_rounded))


def edge_count(adj: np.ndarray) -> int:
    return int(np.sum(adj) // 2)


def laplacian(adj: np.ndarray) -> np.ndarray:
    # Cast to signed/float before subtraction.
    # If we keep uint8, off-diagonal -1 entries underflow to huge positive values.
    adj_f = adj.astype(np.float64, copy=False)
    deg = np.sum(adj_f, axis=1, dtype=np.float64)
    return np.diag(deg) - adj_f


def spectral_features(adj: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    L = laplacian(adj).astype(np.float64)
    evals, evecs = np.linalg.eigh(L)

    n = adj.shape[0]
    if n >= 3:
        lambda2 = float(evals[1])
        lambda3 = float(evals[2])
        phi2 = evecs[:, 1]
        phi3 = evecs[:, 2]
    elif n == 2:
        lambda2 = float(evals[1])
        lambda3 = float(evals[1])
        phi2 = evecs[:, 1]
        phi3 = evecs[:, 1]
    else:
        lambda2 = 0.0
        lambda3 = 0.0
        phi2 = np.zeros((n,), dtype=np.float64)
        phi3 = np.zeros((n,), dtype=np.float64)

    return lambda2, lambda3, phi2.astype(np.float64), phi3.astype(np.float64)


def algebraic_connectivity(adj: np.ndarray) -> float:
    """Score a graph without computing unused eigenvectors or the full spectrum."""
    if len(adj) < 2:
        return 0.0
    matrix = laplacian(adj)
    if scipy_eigvalsh is not None:
        return float(scipy_eigvalsh(matrix, subset_by_index=(1, 1), driver="evr")[0])
    return float(np.linalg.eigvalsh(matrix)[1])


def node_degrees(adj: np.ndarray) -> np.ndarray:
    return np.sum(adj, axis=1).astype(np.float64)


def two_hop_neighbor_counts(adj: np.ndarray) -> np.ndarray:
    # Avoid uint8 overflow in adjacency multiplication.
    adj_bool = adj > 0
    adj_i = adj_bool.astype(np.int64, copy=False)
    a2 = (adj_i @ adj_i) > 0
    np.fill_diagonal(a2, False)
    two_hop_only = np.logical_and(a2, np.logical_not(adj_bool))
    return np.sum(two_hop_only, axis=1, dtype=np.float64)


def clustering_coefficients(adj: np.ndarray) -> np.ndarray:
    adj_f = adj.astype(np.float64, copy=False)
    deg = np.sum(adj_f, axis=1, dtype=np.float64)
    # For simple undirected graphs: (A^3)_{ii} = 2 * (# triangles touching node i).
    a3_diag = np.diag(adj_f @ adj_f @ adj_f)
    denom = deg * np.maximum(1.0, deg - 1.0)
    coeffs = np.divide(
        a3_diag,
        denom,
        out=np.zeros_like(a3_diag, dtype=np.float64),
        where=deg >= 2.0,
    )
    return coeffs


def non_edges(adj: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.triu(adj == 0, k=1)
    rows, cols = np.where(mask)
    return list(zip(rows.tolist(), cols.tolist()))


def edge_index_from_adj(adj: np.ndarray) -> np.ndarray:
    rows, cols = np.where(adj > 0)
    return np.vstack([rows, cols]).astype(np.int64)


def truncated_all_pairs_shortest_path(adj: np.ndarray, cap: int) -> np.ndarray:
    if scipy_shortest_path is not None:
        dist = scipy_shortest_path(
            adj.astype(np.float64, copy=False),
            directed=False,
            unweighted=True,
        )
        dist = np.minimum(dist, float(cap))
        return dist.astype(np.int64, copy=False)

    n = adj.shape[0]
    dist = np.full((n, n), fill_value=cap + 1, dtype=np.int64)
    for src in range(n):
        dist[src, src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            if dist[src, u] >= cap:
                continue
            neighbors = np.where(adj[u] > 0)[0]
            for v in neighbors:
                if dist[src, v] > dist[src, u] + 1:
                    dist[src, v] = dist[src, u] + 1
                    q.append(v)
    dist = np.minimum(dist, cap)
    return dist


def jaccard_overlap(adj: np.ndarray, i: int, j: int) -> float:
    ni = set(np.where(adj[i] > 0)[0].tolist())
    nj = set(np.where(adj[j] > 0)[0].tolist())
    union = ni.union(nj)
    if not union:
        return 0.0
    inter = ni.intersection(nj)
    return float(len(inter)) / float(len(union))


def common_neighbors_norm(adj: np.ndarray, i: int, j: int) -> float:
    n = adj.shape[0]
    denom = max(1, n - 2)
    common = np.sum((adj[i] > 0) & (adj[j] > 0))
    return float(common) / float(denom)


def fiedler_scores_for_pairs(phi2: np.ndarray, pairs: List[Tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return np.zeros((0,), dtype=np.float64)
    arr = np.asarray(pairs, dtype=np.int64)
    diff = phi2[arr[:, 0]] - phi2[arr[:, 1]]
    return diff * diff


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if len(scores) <= k:
        return np.arange(len(scores), dtype=np.int64)
    order = np.argsort(-scores, kind="mergesort")
    return order[:k].astype(np.int64)
