from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import random
import unittest

import networkx as nx
import numpy as np

from graph_design.backbones.cayley import (
    CayleyBackboneGenerator,
    _InverseClosedSampler,
    _LiftDeleteBatchEvaluator,
    _SemidihedralNoDeleteCharacteristicEvaluator,
    _build_batch_evaluator,
    _build_cayley_graph,
    _build_group,
    _split_cosets,
)
from graph_design.config import DesignConfig
from graph_design.types import DesignProblem


class SemidihedralSpectralTests(unittest.TestCase):
    def test_all_inverse_closed_subsets_of_sd16(self) -> None:
        group = _build_group(target_n=16, index=2, lift_nodes=0)
        sampler = _InverseClosedSampler(group, list(range(1, group.order)))
        atoms = [(g,) for g in sampler.involutions] + sampler.pairs
        by_degree = defaultdict(list)
        for mask in range(1 << len(atoms)):
            generators = sorted(
                g for i, atom in enumerate(atoms) if mask & (1 << i) for g in atom
            )
            by_degree[len(generators)].append(generators)

        fast = _SemidihedralNoDeleteCharacteristicEvaluator(group=group)
        dense = _LiftDeleteBatchEvaluator(group=group, delete_nodes=(), backend="cpu")
        checked = 0
        for degree, generators in by_degree.items():
            with self.subTest(degree=degree):
                matrix = np.asarray(generators, dtype=np.int32)
                actual, edges = fast.lambda2_edges_from_index_matrix(matrix)
                expected, expected_edges = dense.lambda2_edges_from_index_matrix(matrix)
                np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-12)
                np.testing.assert_array_equal(edges, expected_edges)
                checked += len(generators)
        self.assertEqual(checked, 1024)

    def test_random_full_and_coset_generators(self) -> None:
        rng = random.Random(2701)
        for n in (16, 32, 64, 128):
            group = _build_group(target_n=n, index=2, lift_nodes=0)
            fast = _SemidihedralNoDeleteCharacteristicEvaluator(group=group)
            dense = _LiftDeleteBatchEvaluator(group=group, delete_nodes=(), backend="cpu")
            _, outside = _split_cosets(group, 2)
            for pool in (list(range(1, n)), outside):
                sampler = _InverseClosedSampler(group, pool)
                for degree in sorted({1, 2, 3, len(pool) // 2, len(pool)}):
                    if not sampler.feasible(degree):
                        continue
                    with self.subTest(n=n, pool_size=len(pool), degree=degree):
                        matrix = np.asarray(
                            [sorted(sampler.sample(degree, rng)) for _ in range(8)],
                            dtype=np.int32,
                        )
                        actual, edges = fast.lambda2_edges_from_index_matrix(matrix)
                        expected, expected_edges = dense.lambda2_edges_from_index_matrix(matrix)
                        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-12)
                        np.testing.assert_array_equal(edges, expected_edges)
                        if degree == len(pool):
                            np.testing.assert_allclose(actual, actual[0], atol=1e-12)

    def test_independent_graph_laplacian_oracle(self) -> None:
        for n in (16, 32, 64):
            group = _build_group(target_n=n, index=2, lift_nodes=0)
            fast = _SemidihedralNoDeleteCharacteristicEvaluator(group=group)
            cases = (
                {n // 2},  # Disconnected matching, with many zero eigenvalues.
                set(range(1, n // 2)),  # Two disjoint cliques.
                set(range(n // 2, n)),  # Complete bipartite graph.
                set(range(1, n)),  # Complete graph.
                {1, group.inv(1), n // 2, n // 2 + 1, group.inv(n // 2 + 1)},
            )
            for generators in cases:
                with self.subTest(n=n, generators=sorted(generators)):
                    graph = _build_cayley_graph(group, generators)
                    adjacency = nx.to_numpy_array(graph, nodelist=range(n))
                    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
                    expected = np.linalg.eigvalsh(laplacian)[1]
                    actual, edges = fast.lambda2_edges_from_index_matrix(
                        np.asarray([sorted(generators)], dtype=np.int32)
                    )
                    self.assertAlmostEqual(float(actual[0]), float(expected), places=10)
                    self.assertEqual(int(edges[0]), graph.number_of_edges())

    def test_empty_batches_and_invalid_inputs(self) -> None:
        group = _build_group(target_n=16, index=2, lift_nodes=0)
        fast = _SemidihedralNoDeleteCharacteristicEvaluator(group=group)
        for shape in ((0, 3), (0, 0), (2, 0)):
            values, edges = fast.lambda2_edges_from_index_matrix(np.empty(shape, dtype=np.int32))
            np.testing.assert_array_equal(values, np.zeros(shape[0]))
            np.testing.assert_array_equal(edges, np.zeros(shape[0], dtype=np.int64))
        for matrix in (np.array([1, 2]), np.array([[0]]), np.array([[-1]]), np.array([[16]]), np.array([[1]])):
            with self.subTest(matrix=matrix.tolist()), self.assertRaises(ValueError):
                fast.lambda2_edges_from_index_matrix(matrix)
        for invalid in (replace(group, order=15), replace(group, half=0), replace(group, twist=7)):
            with self.assertRaises(ValueError):
                _SemidihedralNoDeleteCharacteristicEvaluator(group=invalid)

    def test_dispatch_and_deletion_fallback(self) -> None:
        config = DesignConfig(cayley_eval_mode="character")
        for n, index, lift, backend in (
            (16, 2, 0, "character_semidihedral_no_delete"),
            (32, 2, 0, "character_semidihedral_no_delete"),
            (18, 2, 0, "character_dihedral_no_delete"),
            (18, 3, 0, "character_cyclic_no_delete"),
            (17, 2, 1, "dense_laplacian"),
            (16, 3, 2, "dense_laplacian"),
        ):
            with self.subTest(n=n, index=index):
                group = _build_group(target_n=n, index=index, lift_nodes=lift)
                _, actual = _build_batch_evaluator(
                    group=group, delete_nodes=tuple(range(lift)), config=config
                )
                self.assertEqual(actual, backend)

        group = _build_group(target_n=16, index=2, lift_nodes=0)
        for config, deleted in ((DesignConfig(), ()), (config, (0,))):
            evaluator, backend = _build_batch_evaluator(group=group, delete_nodes=deleted, config=config)
            self.assertIsInstance(evaluator, _LiftDeleteBatchEvaluator)
            self.assertEqual(backend, "dense_laplacian")
            expected = _LiftDeleteBatchEvaluator(group=group, delete_nodes=deleted, backend="cpu")
            matrix = np.asarray([list(range(1, 16))], dtype=np.int32)
            actual_values, actual_edges = evaluator.lambda2_edges_from_index_matrix(matrix)
            expected_values, expected_edges = expected.lambda2_edges_from_index_matrix(matrix)
            np.testing.assert_allclose(actual_values, expected_values)
            np.testing.assert_array_equal(actual_edges, expected_edges)

    def test_backbone_selection_quality_with_identical_sampling(self) -> None:
        generator = CayleyBackboneGenerator()
        config = DesignConfig(cayley_samples=40, cayley_eval_batch_size=16)
        for n in (16, 32, 64):
            for degree in (3, n // 4, n // 2):
                for seed in (0, 1):
                    with self.subTest(n=n, degree=degree, seed=seed):
                        problem = DesignProblem(n=n, m=n * degree // 2)
                        dense = generator.generate(problem, config, random.Random(seed))
                        fast = generator.generate(
                            problem, replace(config, cayley_eval_mode="character"), random.Random(seed)
                        )
                        self.assertIsNotNone(dense)
                        self.assertIsNotNone(fast)
                        self.assertEqual(fast.metadata["eval_backend"], "character_semidihedral_no_delete")
                        self.assertAlmostEqual(dense.backbone_lambda2, fast.backbone_lambda2, places=10)
                        self.assertEqual(dense.graph.number_of_edges(), fast.graph.number_of_edges())
                        self.assertEqual(dense.metadata["sample_count_unique"], fast.metadata["sample_count_unique"])
                        self.assertTrue(nx.is_connected(fast.graph))


if __name__ == "__main__":
    unittest.main()
