from __future__ import annotations

from collections import deque
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from rl_train.config import load_config
from rl_train.curriculum import CurriculumScheduler
from rl_train.env import GraphEnv
from rl_train.features.candidates import _stable_topk_indices, build_candidate_features
from rl_train.graph_math import algebraic_connectivity, laplacian, normalized_density, spectral_features
from rl_train.model import BBPolicyNetwork


def reference_distances(adj):
    n = len(adj)
    result = np.full((n, n), n + 1, dtype=np.int64)
    for src in range(n):
        result[src, src] = 0
        pending = deque([src])
        while pending:
            u = pending.popleft()
            for v in np.flatnonzero(adj[u]):
                if result[src, v] > result[src, u] + 1:
                    result[src, v] = result[src, u] + 1
                    pending.append(v)
    return result


def make_env(*, fast=False, variant="lite_v3"):
    cfg = load_config("configs/bbrl.yaml")
    return GraphEnv(
        env_id=0, scheduler=CurriculumScheduler(cfg.curriculum), top_k=32,
        dist_cap=4, terminal_bonus_coef=0.2, seed=5, rl_variant=variant,
        compute_spectral_each_step=False, inference_only=fast,
    )


class InferenceTests(unittest.TestCase):

    def test_distances_match_bfs_including_disconnected_and_fallback(self):
        env = make_env()
        rng = np.random.default_rng(71)
        for n in (1, 2, 9, 33):
            for density in (0.0, 0.05, 0.5, 1.0):
                upper = np.triu((rng.random((n, n)) < density).astype(np.uint8), 1)
                adj = upper + upper.T
                expected = reference_distances(adj)
                np.testing.assert_array_equal(env._compute_all_pairs_shortest_path(adj), expected)
                with patch("rl_train.graph_math.scipy_shortest_path", None):
                    np.testing.assert_array_equal(env._compute_all_pairs_shortest_path(adj), expected)

    def test_incremental_caches_match_integer_reference_after_each_edge(self):
        rng = np.random.default_rng(19)
        env = make_env()
        for n, density in ((12, 0.0), (33, 0.15), (260, 0.99)):
            upper = np.triu((rng.random((n, n)) < density).astype(np.uint8), 1)
            env.reset_with_target(n, 1.0, initial_adj=upper + upper.T)
            for _ in range(5):
                a = env.adj.astype(np.int64)
                a2 = a @ a
                np.testing.assert_array_equal(env._a2_counts, a2)
                np.testing.assert_array_equal(env._triangles_per_node, np.sum(a2 * a, axis=1) / 2)
                np.testing.assert_array_equal(env._dist_matrix, reference_distances(a))
                pairs = np.argwhere(np.triu(a == 0, 1))
                if not len(pairs):
                    break
                env.step(tuple(int(x) for x in pairs[rng.integers(len(pairs))]))

    def test_partial_selection_preserves_stable_ties(self):
        rng = np.random.default_rng(32)
        for size in (0, 1, 32, 101, 1000):
            for scores in (rng.normal(size=size), rng.integers(-3, 4, size=size).astype(float), np.ones(size)):
                for k in (0, 1, 7, 32, size, size + 1):
                    expected = np.argsort(-scores, kind="mergesort")[:k]
                    np.testing.assert_array_equal(_stable_topk_indices(scores, k), expected)
        special = np.array([1.0, np.nan, np.inf, -np.inf, 1.0])
        np.testing.assert_array_equal(_stable_topk_indices(special, 2), np.argsort(-special, kind="mergesort")[:2])

    def test_scoring_matches_dense_for_repeated_and_zero_eigenvalues(self):
        for n in (1, 2, 16, 33):
            for adj in (np.zeros((n, n), dtype=np.uint8), np.ones((n, n), dtype=np.uint8) - np.eye(n, dtype=np.uint8)):
                expected = float(np.linalg.eigvalsh(laplacian(adj))[1]) if n > 1 else 0.0
                self.assertAlmostEqual(algebraic_connectivity(adj), expected, places=9)
                with patch("rl_train.graph_math.scipy_eigvalsh", None):
                    self.assertAlmostEqual(algebraic_connectivity(adj), expected, places=9)

    def test_fast_observations_and_actor_match_training_decisions(self):
        torch.manual_seed(4)
        model = BBPolicyNetwork(
            node_feature_dim=3, pair_feature_dim=5, global_feature_dim=3,
            scorer_hidden_dim=64, scorer_layers=2, value_hidden_dim=64,
        ).eval()
        old, fast = make_env(), make_env(fast=True)
        for n, m in ((12, 22), (33, 81)):
            old_obs = old.reset_with_target(n, normalized_density(n, m))
            with patch("rl_train.env.spectral_features", side_effect=AssertionError("Unused eigenvectors")):
                fast_obs = fast.reset_with_target(n, normalized_density(n, m))
                done = False
                while not done:
                    for name in ("node_features", "pair_features", "global_features", "candidate_pairs"):
                        np.testing.assert_array_equal(getattr(old_obs, name), getattr(fast_obs, name))
                    self.assertEqual(fast_obs.edge_index.shape, (2, 0))
                    with torch.no_grad():
                        old_output = model.forward_observation(old_obs, torch.device("cpu"))
                    with torch.inference_mode(), patch.object(model.value_mlp, "forward", side_effect=AssertionError("Unused critic")):
                        output = model.forward_observation(fast_obs, torch.device("cpu"), compute_value=False)
                    self.assertTrue(torch.equal(output.logits, old_output.logits))
                    pair = tuple(int(v) for v in fast_obs.candidate_pairs[torch.argmax(output.logits).item()])
                    # The legacy terminal path still computes eigenvectors.
                    with patch("rl_train.env.spectral_features", wraps=spectral_features):
                        old_obs, _, old_done, old_info = old.step(pair)
                    fast_obs, _, done, info = fast.step(pair)
                    self.assertEqual(done, old_done)
                np.testing.assert_array_equal(fast.adj, old.adj)
                self.assertEqual(fast_obs.candidate_pairs.shape, (0, 2))
                self.assertAlmostEqual(info["terminal_lambda2_norm"], old_info["terminal_lambda2_norm"], places=10)

    def test_zero_edge_reset_skips_features_and_state_roundtrip(self):
        env = make_env(fast=True)
        adj = np.ones((12, 12), dtype=np.uint8) - np.eye(12, dtype=np.uint8)
        with patch.object(env, "_compute_all_pairs_shortest_path", side_effect=AssertionError("Unused APSP")), patch.object(env, "_build_observation", side_effect=AssertionError("Unused features")):
            obs = env.reset_with_target(12, 1.0, initial_adj=adj)
        self.assertAlmostEqual(obs.lambda2_norm, 1.0)
        self.assertIsNone(env._a2_counts)
        restored = make_env()
        restored.load_state_dict(env.state_dict())
        self.assertTrue(restored.inference_only)
        np.testing.assert_array_equal(restored.adj, adj)
        legacy_state = env.state_dict()
        del legacy_state["inference_only"]
        restored.load_state_dict(legacy_state)
        self.assertFalse(restored.inference_only)

    def test_feature_cache_matches_uncached_and_apsp_is_computed_once(self):
        env = make_env()
        obs = env.reset_with_target(33, 0.4)
        kwargs = dict(adj=env.adj, deg=env._deg_cache, node_features=obs.node_features,
                      n=env.n, dist_cap=env.dist_cap, incremental_observation=True,
                      pair_i=env._pair_i, pair_j=env._pair_j, pair_is_nonedge=env._pair_is_nonedge,
                      dist_matrix=env._dist_matrix, node_budget=32, pair_budget=96, pair_budget_min=32)
        reference = build_candidate_features(**kwargs)
        cached = build_candidate_features(**kwargs, a2_counts=env._a2_counts)
        for a, b in zip(reference, cached):
            np.testing.assert_array_equal(a, b)
        kwargs["dist_matrix"] = None
        from rl_train.features import candidates
        with patch.object(candidates, "truncated_all_pairs_shortest_path", wraps=candidates.truncated_all_pairs_shortest_path) as distance:
            build_candidate_features(**kwargs)
        self.assertEqual(distance.call_count, 1)


if __name__ == "__main__":
    unittest.main()
