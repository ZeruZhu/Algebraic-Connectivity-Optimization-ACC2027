from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..graph_math import non_edges, truncated_all_pairs_shortest_path


def _stable_topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if k <= 0 or scores.size == 0:
        return np.zeros((0,), dtype=np.int64)
    if k * 2 >= scores.size or not np.all(np.isfinite(scores)):
        return np.argsort(-scores, kind="mergesort")[:k].astype(np.int64, copy=False)
    # Resolve cutoff ties in original index order, exactly as stable full sorting.
    cutoff = np.partition(scores, scores.size - k)[scores.size - k]
    above = np.flatnonzero(scores > cutoff)
    tied = np.flatnonzero(scores == cutoff)[: k - above.size]
    selected = np.sort(np.concatenate((above, tied)))
    return selected[np.argsort(-scores[selected], kind="mergesort")].astype(np.int64, copy=False)


def _node_scores(node_features: np.ndarray) -> np.ndarray:
    # node_features columns for BB-RL node features:
    # [deg_norm, two_hop_norm, clustering]
    if node_features.size == 0:
        return np.zeros((0,), dtype=np.float64)
    nf = np.asarray(node_features, dtype=np.float64)
    return (0.55 * nf[:, 0]) + (0.30 * nf[:, 1]) + (0.15 * nf[:, 2])


def build_candidate_features(
    *,
    adj: np.ndarray,
    deg: np.ndarray,
    node_features: np.ndarray,
    n: int,
    dist_cap: int,
    incremental_observation: bool,
    pair_i: Optional[np.ndarray],
    pair_j: Optional[np.ndarray],
    pair_is_nonedge: Optional[np.ndarray],
    dist_matrix: Optional[np.ndarray],
    node_budget: int,
    pair_budget: int,
    pair_budget_min: int,
    a2_counts: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    if (
        incremental_observation
        and pair_i is not None
        and pair_j is not None
        and pair_is_nonedge is not None
    ):
        nonedge_pair_idx = np.flatnonzero(pair_is_nonedge).astype(np.int64, copy=False)
        p_i = pair_i
        p_j = pair_j
    else:
        all_non_edges = non_edges(adj)
        if not all_non_edges:
            return (
                np.zeros((0, 2), dtype=np.int64),
                np.zeros((0, 5), dtype=np.float32),
                0,
                0,
            )
        pairs = np.asarray(all_non_edges, dtype=np.int64)
        p_i = pairs[:, 0]
        p_j = pairs[:, 1]
        nonedge_pair_idx = np.arange(pairs.shape[0], dtype=np.int64)

    raw_count = int(nonedge_pair_idx.size)
    if raw_count == 0:
        return (
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0, 5), dtype=np.float32),
            0,
            0,
        )

    if dist_matrix is None:
        dist_matrix = truncated_all_pairs_shortest_path(adj, dist_cap)

    node_budget_i = max(2, min(int(node_budget), int(n)))
    pair_budget_i = max(1, int(pair_budget))
    pair_budget_min_i = max(1, int(pair_budget_min))
    pair_budget_cap = max(pair_budget_i, pair_budget_min_i)
    target_cap = min(raw_count, pair_budget_cap)

    n_scores = _node_scores(node_features)
    top_nodes_idx = _stable_topk_indices(n_scores, node_budget_i)
    top_nodes = np.zeros((n,), dtype=bool)
    top_nodes[top_nodes_idx] = True

    nonedge_i = p_i[nonedge_pair_idx]
    nonedge_j = p_j[nonedge_pair_idx]
    in_top = np.logical_and(top_nodes[nonedge_i], top_nodes[nonedge_j])
    pool_idx = nonedge_pair_idx[np.flatnonzero(in_top)]

    if pool_idx.size < pair_budget_min_i:
        # Deterministic fallback: supplement from global non-edge pool.
        needed = int(min(raw_count, pair_budget_min_i) - pool_idx.size)
        if needed > 0:
            pool_mask = np.isin(nonedge_pair_idx, pool_idx, assume_unique=False)
            remain_local = np.flatnonzero(~pool_mask)
            if remain_local.size > 0:
                remain_idx = nonedge_pair_idx[remain_local]
                rem_i = p_i[remain_idx]
                rem_j = p_j[remain_idx]
                denom = float(max(1, n - 1))
                deg_sum_norm = (deg[rem_i] + deg[rem_j]) / (2.0 * denom)
                deg_gap_norm = np.abs(deg[rem_i] - deg[rem_j]) / denom
                rem_score = deg_sum_norm - (0.25 * deg_gap_norm)
                add_local = _stable_topk_indices(rem_score, min(needed, remain_idx.size))
                pool_idx = np.concatenate([pool_idx, remain_idx[add_local]])

    if pool_idx.size == 0:
        # Final safety fallback.
        pool_idx = nonedge_pair_idx.copy()

    pool_i = p_i[pool_idx]
    pool_j = p_j[pool_idx]
    denom = float(max(1, n - 1))
    deg_sum_norm_pool = (deg[pool_i] + deg[pool_j]) / (2.0 * denom)
    deg_gap_norm_pool = np.abs(deg[pool_i] - deg[pool_j]) / denom
    if dist_matrix is not None:
        dist_norm_pool = (
            np.minimum(dist_matrix[pool_i, pool_j], dist_cap).astype(np.float64)
            / float(max(1, dist_cap))
        )
    else:
        all_dist = truncated_all_pairs_shortest_path(adj, dist_cap)
        dist_norm_pool = all_dist[pool_i, pool_j].astype(np.float64) / float(max(1, dist_cap))
    pre_score = (0.70 * deg_sum_norm_pool) - (0.20 * deg_gap_norm_pool) + (0.10 * (1.0 - dist_norm_pool))

    take_local = _stable_topk_indices(pre_score, target_cap)
    selected_idx = pool_idx[take_local]

    if selected_idx.size < min(raw_count, pair_budget_min_i):
        needed = int(min(raw_count, pair_budget_min_i) - selected_idx.size)
        if needed > 0:
            selected_mask = np.isin(nonedge_pair_idx, selected_idx, assume_unique=False)
            remain_local = np.flatnonzero(~selected_mask)
            if remain_local.size > 0:
                remain_idx = nonedge_pair_idx[remain_local]
                rem_i = p_i[remain_idx]
                rem_j = p_j[remain_idx]
                rem_deg_sum = (deg[rem_i] + deg[rem_j]) / (2.0 * denom)
                rem_deg_gap = np.abs(deg[rem_i] - deg[rem_j]) / denom
                rem_score = rem_deg_sum - (0.25 * rem_deg_gap)
                add_local = _stable_topk_indices(rem_score, min(needed, remain_idx.size))
                selected_idx = np.concatenate([selected_idx, remain_idx[add_local]])

    if selected_idx.size > target_cap:
        sel_i = p_i[selected_idx]
        sel_j = p_j[selected_idx]
        sel_deg_sum = (deg[sel_i] + deg[sel_j]) / (2.0 * denom)
        sel_deg_gap = np.abs(deg[sel_i] - deg[sel_j]) / denom
        sel_score = sel_deg_sum - (0.25 * sel_deg_gap)
        keep_local = _stable_topk_indices(sel_score, target_cap)
        selected_idx = selected_idx[keep_local]

    cand_i = p_i[selected_idx]
    cand_j = p_j[selected_idx]
    candidate_pairs = np.stack([cand_i, cand_j], axis=1).astype(np.int64, copy=False)

    if incremental_observation and a2_counts is not None:
        common_counts = a2_counts[cand_i, cand_j].astype(np.float64)
    else:
        common_counts = np.sum(
            np.logical_and(adj[cand_i] > 0, adj[cand_j] > 0),
            axis=1,
            dtype=np.float64,
        )
    cn_norm = common_counts / float(max(1, n - 2))

    union = deg[cand_i] + deg[cand_j] - common_counts
    jac = np.divide(
        common_counts,
        union,
        out=np.zeros_like(common_counts, dtype=np.float64),
        where=union > 0,
    )

    if dist_matrix is not None:
        clipped_dist = np.minimum(dist_matrix[cand_i, cand_j], dist_cap)
        dist_norm = clipped_dist.astype(np.float64) / float(max(1, dist_cap))
    else:
        all_dist = truncated_all_pairs_shortest_path(adj, dist_cap)
        dist_norm = all_dist[cand_i, cand_j].astype(np.float64) / float(max(1, dist_cap))

    deg_sum_norm = (deg[cand_i] + deg[cand_j]) / (2.0 * denom)
    deg_gap_norm = np.abs(deg[cand_i] - deg[cand_j]) / denom

    pair_features = np.stack(
        [cn_norm, jac, dist_norm, deg_sum_norm, deg_gap_norm],
        axis=1,
    ).astype(np.float32, copy=False)

    selected_count = int(candidate_pairs.shape[0])
    return candidate_pairs, pair_features, raw_count, selected_count
