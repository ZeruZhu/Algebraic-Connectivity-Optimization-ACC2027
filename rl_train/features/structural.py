from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..graph_math import (
    clustering_coefficients,
    edge_index_from_adj,
    node_degrees,
    non_edges,
    truncated_all_pairs_shortest_path,
    two_hop_neighbor_counts,
)

_LOG64 = float(np.log(64.0))


def build_node_features(
    *,
    adj: np.ndarray,
    n: int,
    incremental_observation: bool,
    deg_cache: Optional[np.ndarray],
    a2_counts: Optional[np.ndarray],
    triangles_per_node: Optional[np.ndarray],
    eye_mask: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    if (
        incremental_observation
        and deg_cache is not None
        and a2_counts is not None
        and triangles_per_node is not None
        and eye_mask is not None
    ):
        deg = deg_cache
        adj_bool = adj > 0
        two_hop_mask = (a2_counts > 0) & (~adj_bool) & (~eye_mask)
        two_hop = np.sum(two_hop_mask, axis=1, dtype=np.float64)
        cluster_denom = deg * np.maximum(1.0, deg - 1.0)
        cluster = np.divide(
            2.0 * triangles_per_node,
            cluster_denom,
            out=np.zeros_like(deg, dtype=np.float64),
            where=deg >= 2.0,
        )
    else:
        deg = node_degrees(adj)
        two_hop = two_hop_neighbor_counts(adj)
        cluster = clustering_coefficients(adj)

    denom = max(1, n - 1)
    node_features = np.stack(
        [
            deg / denom,
            two_hop / denom,
            cluster,
        ],
        axis=1,
    ).astype(np.float32)
    return node_features, deg


def build_global_features(*, n: int, rho_target: float, rho_current: float) -> np.ndarray:
    n_log_norm = float(np.log(float(max(2, n))) / _LOG64)
    return np.array(
        [
            n_log_norm,
            float(rho_target),
            float(rho_current),
        ],
        dtype=np.float32,
    )




def build_edge_index(adj: np.ndarray) -> np.ndarray:
    return edge_index_from_adj(adj)
