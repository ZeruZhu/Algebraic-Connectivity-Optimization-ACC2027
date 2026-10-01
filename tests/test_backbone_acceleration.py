from dataclasses import replace
import random
from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest
from scipy.sparse.linalg import ArpackNoConvergence

from graph_design.backbones import cayley
from graph_design.backbones.cayley_fft_delete import CyclicDeleteFFTEvaluator
from graph_design.config import DesignConfig
from graph_design.types import DesignProblem


def oracle(group, generators, deleted=()):
    graph = nx.Graph()
    graph.add_nodes_from(range(group.order))
    for g in range(group.order):
        for s in generators:
            h = group.mul(g, s)
            if g != h:
                graph.add_edge(g, h)
    graph.remove_nodes_from(deleted)
    adj = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes()))
    return adj, np.diag(adj.sum(axis=1)) - adj


@pytest.mark.parametrize("group", [cayley._CyclicGroup(15), cayley._build_group(16, 2, 0), cayley._build_group(18, 2, 0)])
def test_vectorized_group_tables_and_graphs(group):
    nodes = np.arange(group.order)
    actions = cayley._right_actions(group, nodes, nodes)
    expected = np.asarray([[group.mul(g, s) for g in nodes] for s in nodes])
    np.testing.assert_array_equal(actions, expected)
    table = cayley._adjacency_generator_table(group)
    assert not table.flags.writeable
    assert cayley._adjacency_generator_table(group) is table
    for generators in [set(), {1}, set(range(1, group.order)), {1, group.inv(1)}]:
        graph = cayley._build_cayley_graph(group, generators)
        adj, _ = oracle(group, generators)
        np.testing.assert_array_equal(nx.to_numpy_array(graph, nodelist=nodes), adj)


def test_feasibility_cache_preserves_rng_and_sampling():
    group = cayley._build_group(32, 2, 0)
    sampler = cayley._InverseClosedSampler(group, list(range(1, 32)))
    first, second = random.Random(701), random.Random(701)
    for degree in [1, 2, 8, 16, 30, 31] * 20:
        feasible = [t for t in range(len(sampler.involutions) + 1)
                    if 0 <= degree - t <= 2 * len(sampler.pairs) and (degree - t) % 2 == 0]
        t = first.choice(feasible)
        expected = set(first.sample(sampler.involutions, t))
        for pair in first.sample(sampler.pairs, (degree - t) // 2):
            expected.update(pair)
        assert sampler.sample(degree, second) == expected
    assert first.getstate() == second.getstate()
    assert len(sampler._feasible_counts) == 6


@pytest.mark.parametrize("group,deleted", [
    (cayley._CyclicGroup(9), (0,)), (cayley._CyclicGroup(12), (1, 5)),
    (cayley._build_group(16, 2, 0), (0, 3)),
    (cayley._build_group(18, 2, 0), (2,)),
])
def test_dense_subset_and_assembly_match_independent_oracle(group, deleted):
    sampler = cayley._InverseClosedSampler(group, list(range(1, group.order)))
    rng = random.Random(32)
    evaluators = [cayley._LiftDeleteBatchEvaluator(group=group, delete_nodes=deleted, backend="cpu", solver=s)
                  for s in ("full", "subset")]
    for degree in range(1, group.order):
        if not sampler.feasible(degree):
            continue
        matrix = np.asarray([sorted(sampler.sample(degree, rng)) for _ in range(3)])
        expected = [oracle(group, row, deleted) for row in matrix]
        for evaluator in evaluators:
            values, edges = evaluator.lambda2_edges_from_index_matrix(matrix)
            np.testing.assert_allclose(values, [np.linalg.eigvalsh(lap)[1] for _, lap in expected], atol=1e-10)
            np.testing.assert_array_equal(edges, [adj.sum() // 2 for adj, _ in expected])


@pytest.mark.parametrize("order,deleted", [(9, (0,)), (12, (1, 5)), (6, (0, 1, 2, 3))])
def test_fft_all_inverse_closed_sets_and_operator(order, deleted):
    group = cayley._CyclicGroup(order)
    sampler = cayley._InverseClosedSampler(group, list(range(1, order)))
    atoms = [(g,) for g in sampler.involutions] + sampler.pairs
    evaluator = CyclicDeleteFFTEvaluator(order=order, delete_nodes=deleted)
    for mask in range(1 << len(atoms)):
        generators = sorted(g for i, atom in enumerate(atoms) if mask & (1 << i) for g in atom)
        adj, lap = oracle(group, generators, deleted)
        values, edges = evaluator.lambda2_edges_from_index_matrix(np.asarray([generators], dtype=np.int32))
        np.testing.assert_allclose(values, [np.linalg.eigvalsh(lap)[1]], atol=1e-9)
        assert edges[0] == adj.sum() // 2
        indicator = np.zeros(order)
        indicator[generators] = 1
        operator, degree = evaluator._operator(indicator)
        vector = np.arange(evaluator.keep.size, dtype=float)
        shift = 2 * degree.max() + 1
        expected = shift * (vector - vector.mean()) - lap @ vector
        np.testing.assert_allclose(operator @ vector, expected, atol=1e-11)
        np.testing.assert_allclose(operator @ vector[:, None], expected[:, None], atol=1e-11)


def test_fft_fallback_and_input_checks():
    evaluator = CyclicDeleteFFTEvaluator(order=9, delete_nodes=(0,))
    matrix = np.asarray([[1, 8]])
    _, lap = oracle(cayley._CyclicGroup(9), matrix[0], (0,))
    failure = ArpackNoConvergence("forced", np.empty(0), np.empty((8, 0)))
    with patch("graph_design.backbones.cayley_fft_delete.eigsh", side_effect=failure) as solver:
        values, _ = evaluator.lambda2_edges_from_index_matrix(matrix)
    assert solver.call_args.kwargs['maxiter'] == 300
    assert solver.call_args.kwargs['which'] == 'LA'
    assert evaluator.fallback_count == 1
    np.testing.assert_allclose(values, [np.linalg.eigvalsh(lap)[1]], atol=1e-11)
    with patch("graph_design.backbones.cayley_fft_delete.eigsh", return_value=(np.asarray([99.0]), np.ones((8, 1)))):
        values, _ = evaluator.lambda2_edges_from_index_matrix(matrix)
    assert evaluator.fallback_count == 2
    np.testing.assert_allclose(values, [np.linalg.eigvalsh(lap)[1]], atol=1e-11)
    for shape in ((0, 2), (2, 0)):
        values, edges = evaluator.lambda2_edges_from_index_matrix(np.empty(shape, dtype=np.int32))
        np.testing.assert_array_equal(values, np.zeros(shape[0]))
        np.testing.assert_array_equal(edges, np.zeros(shape[0]))
    for invalid in [np.array([1]), np.array([[0]]), np.array([[9]]), np.array([[1]])]:
        with pytest.raises(ValueError):
            evaluator.lambda2_edges_from_index_matrix(invalid)
    for deleted in [(-1,), (9,), tuple(range(8))]:
        with pytest.raises(ValueError):
            CyclicDeleteFFTEvaluator(order=9, delete_nodes=deleted)


def test_fft_disconnected_parent_shortcut_keeps_deletion_exception():
    matrix = np.asarray([[2, 4]])
    disconnected = CyclicDeleteFFTEvaluator(order=6, delete_nodes=(0,))
    with patch('graph_design.backbones.cayley_fft_delete.eigsh', side_effect=AssertionError('must not solve')):
        values, edges = disconnected.lambda2_edges_from_index_matrix(matrix)
    assert values[0] == 0 and edges[0] == 4
    assert disconnected.disconnected_shortcuts == 1
    connected = CyclicDeleteFFTEvaluator(order=6, delete_nodes=(1, 3, 5))
    values, edges = connected.lambda2_edges_from_index_matrix(matrix)
    np.testing.assert_allclose(values, [3.0], atol=1e-11)
    assert edges[0] == 3 and connected.disconnected_shortcuts == 0


def test_solver_dispatch_and_selection_quality():
    group = cayley._CyclicGroup(33)
    config = DesignConfig(cayley_eval_mode="character", cayley_samples=40,
                          cayley_multi_index_overlap=True, cayley_independent_index_seeds=True)
    for solver, expected in [("full", "dense_laplacian"), ("subset", "dense_subset_laplacian"),
                             ("fft_iterative", "fft_iterative_cyclic_delete")]:
        _, backend = cayley._build_batch_evaluator(group=group, delete_nodes=(0,), config=replace(config, cayley_dense_solver=solver))
        assert backend == expected
    problem = DesignProblem(n=32, m=310)
    candidates = []
    for solver in ("full", "subset", "fft_iterative"):
        candidate = cayley.CayleyBackboneGenerator().generate(problem, replace(config, cayley_dense_solver=solver), random.Random(17))
        assert candidate is not None
        assert candidate.metadata["sample_count_requested"] == 40
        adj = nx.to_numpy_array(candidate.graph, nodelist=range(32))
        score = np.linalg.eigvalsh(np.diag(adj.sum(axis=1)) - adj)[1]
        assert abs(score - candidate.backbone_lambda2) < 1e-9
        candidates.append(candidate)
    np.testing.assert_allclose([c.backbone_lambda2 for c in candidates], candidates[0].backbone_lambda2, atol=1e-9)
    for bad in [replace(config, cayley_dense_solver="bad"), replace(config, cayley_dense_solver="subset", cayley_spectral_backend="cuda")]:
        with pytest.raises(ValueError):
            cayley._build_batch_evaluator(group=group, delete_nodes=(0,), config=bad)
