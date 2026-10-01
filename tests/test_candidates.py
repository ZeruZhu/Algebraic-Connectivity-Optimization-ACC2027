from __future__ import annotations

import unittest

import numpy as np

from rl_train.features.structural import build_node_features
from rl_train.features.candidates import build_candidate_features
from rl_train.graph_math import truncated_all_pairs_shortest_path


def _make_adj(n: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=np.uint8)
    for i in range(n - 1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    extra_edges = [(0, 3), (1, 4), (2, 6), (5, 9), (3, 8)]
    for i, j in extra_edges:
        if i < n and j < n:
            adj[i, j] = 1
            adj[j, i] = 1
    return adj


class CandidateBuilderTests(unittest.TestCase):
    def test_candidate_builder_nonincremental_valid_and_deterministic(self) -> None:
        n = 16
        adj = _make_adj(n)
        node_features, deg = build_node_features(
            adj=adj,
            n=n,
            incremental_observation=False,
            deg_cache=None,
            a2_counts=None,
            triangles_per_node=None,
            eye_mask=None,
        )

        pair_budget = 96
        pair_budget_min = 32
        out_1 = build_candidate_features(
            adj=adj,
            deg=deg,
            node_features=node_features,
            n=n,
            dist_cap=4,
            incremental_observation=False,
            pair_i=None,
            pair_j=None,
            pair_is_nonedge=None,
            dist_matrix=None,
            node_budget=32,
            pair_budget=pair_budget,
            pair_budget_min=pair_budget_min,
        )
        out_2 = build_candidate_features(
            adj=adj,
            deg=deg,
            node_features=node_features,
            n=n,
            dist_cap=4,
            incremental_observation=False,
            pair_i=None,
            pair_j=None,
            pair_is_nonedge=None,
            dist_matrix=None,
            node_budget=32,
            pair_budget=pair_budget,
            pair_budget_min=pair_budget_min,
        )

        candidate_pairs_1, pair_features_1, raw_count_1, selected_count_1 = out_1
        candidate_pairs_2, pair_features_2, raw_count_2, selected_count_2 = out_2

        self.assertEqual(raw_count_1, raw_count_2)
        self.assertEqual(selected_count_1, selected_count_2)
        self.assertTrue(np.array_equal(candidate_pairs_1, candidate_pairs_2))
        self.assertTrue(np.array_equal(pair_features_1, pair_features_2))
        self.assertEqual(candidate_pairs_1.shape[0], selected_count_1)
        self.assertEqual(pair_features_1.shape[0], selected_count_1)
        self.assertEqual(pair_features_1.shape[1], 5)

        expected_min = min(int(raw_count_1), int(pair_budget_min))
        expected_max = min(int(raw_count_1), max(int(pair_budget), int(pair_budget_min)))
        self.assertGreaterEqual(int(selected_count_1), int(expected_min))
        self.assertLessEqual(int(selected_count_1), int(expected_max))

        for i, j in candidate_pairs_1.tolist():
            self.assertNotEqual(i, j)
            self.assertEqual(int(adj[i, j]), 0)
            self.assertEqual(int(adj[j, i]), 0)

    def test_candidate_builder_incremental_matches_nonincremental(self) -> None:
        n = 14
        adj = _make_adj(n)
        node_features, deg = build_node_features(
            adj=adj,
            n=n,
            incremental_observation=False,
            deg_cache=None,
            a2_counts=None,
            triangles_per_node=None,
            eye_mask=None,
        )
        rows, cols = np.triu_indices(n, k=1)
        pair_is_nonedge = (adj[rows, cols] == 0)
        dist_matrix = truncated_all_pairs_shortest_path(adj, 4)

        out_noninc = build_candidate_features(
            adj=adj,
            deg=deg,
            node_features=node_features,
            n=n,
            dist_cap=4,
            incremental_observation=False,
            pair_i=None,
            pair_j=None,
            pair_is_nonedge=None,
            dist_matrix=None,
            node_budget=16,
            pair_budget=48,
            pair_budget_min=20,
        )
        out_inc = build_candidate_features(
            adj=adj,
            deg=deg,
            node_features=node_features,
            n=n,
            dist_cap=4,
            incremental_observation=True,
            pair_i=rows.astype(np.int64, copy=False),
            pair_j=cols.astype(np.int64, copy=False),
            pair_is_nonedge=pair_is_nonedge,
            dist_matrix=dist_matrix,
            node_budget=16,
            pair_budget=48,
            pair_budget_min=20,
        )

        cand_noninc, feat_noninc, raw_noninc, sel_noninc = out_noninc
        cand_inc, feat_inc, raw_inc, sel_inc = out_inc
        self.assertEqual(int(raw_noninc), int(raw_inc))
        self.assertEqual(int(sel_noninc), int(sel_inc))
        self.assertTrue(np.array_equal(cand_noninc, cand_inc))
        self.assertTrue(np.array_equal(feat_noninc, feat_inc))


if __name__ == "__main__":
    unittest.main()
