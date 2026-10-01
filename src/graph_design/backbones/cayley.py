from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import math
import random

import networkx as nx
import numpy as np
from scipy.linalg import eigvalsh as selected_eigvalsh

from graph_design.backbones.base import BackboneGenerator
from graph_design.backbones.cayley_fft_delete import CyclicDeleteFFTEvaluator
from graph_design.config import DesignConfig
from graph_design.types import BackboneCandidate, DesignProblem
from graph_design.utils import relabel_to_integers

try:  # pragma: no cover - optional dependency.
    import cupy as cp
except Exception:  # pragma: no cover - CPU-only environments.
    cp = None


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1) == 0)


def _lift_nodes_needed(n: int, index: int) -> int:
    if index <= 0:
        raise ValueError("index must be positive.")
    return (index - (n % index)) % index


def _index_edge_interval(
    index: int,
    *,
    index2_upper: float = 0.5,
    index3_upper: float = 2.0 / 3.0,
    index4_low: float = 2.0 / 3.0,
) -> tuple[float, float]:
    if index == 2:
        return (0.0, index2_upper)
    if index == 3:
        return (0.5, index3_upper)
    if index == 4:
        return (index4_low, 0.75)
    raise ValueError(f"Unsupported index: {index}")


def _in_open_interval(value: float, low: float, high: float) -> bool:
    return low < value < high


@dataclass(frozen=True)
class _PocketScore:
    index: int
    lift_nodes: int
    lifted_n: int
    k_max: int
    m_proj: int
    edge_ratio: float
    pocket_low: float
    pocket_high: float
    in_pocket: bool


def _score_index_candidate(
    n: int,
    m: int,
    index: int,
    *,
    index2_upper: float = 0.5,
    index3_upper: float = 2.0 / 3.0,
    index4_low: float = 2.0 / 3.0,
) -> _PocketScore:
    j = _lift_nodes_needed(n, index)
    lifted_n = n + j
    denom = max(1, n - j)
    k_max = int((2 * m) // denom)
    k_max = max(1, min(lifted_n - 1, k_max))
    m_proj = int((k_max * denom) // 2)
    edge_ratio = (2.0 * float(m_proj) / float(n * n)) if n > 0 else 0.0
    pocket_low, pocket_high = _index_edge_interval(
        index,
        index2_upper=index2_upper,
        index3_upper=index3_upper,
        index4_low=index4_low,
    )
    in_pocket = _in_open_interval(edge_ratio, pocket_low, pocket_high)
    return _PocketScore(
        index=index,
        lift_nodes=j,
        lifted_n=lifted_n,
        k_max=k_max,
        m_proj=m_proj,
        edge_ratio=edge_ratio,
        pocket_low=pocket_low,
        pocket_high=pocket_high,
        in_pocket=in_pocket,
    )


def _score_index_candidates(
    n: int,
    m: int,
    *,
    index2_upper: float = 0.5,
    index3_upper: float = 2.0 / 3.0,
    index4_low: float = 2.0 / 3.0,
    enable_index4: bool = True,
) -> dict[int, _PocketScore]:
    index_set = (2, 3, 4) if enable_index4 else (2, 3)
    return {
        index: _score_index_candidate(
            n=n,
            m=m,
            index=index,
            index2_upper=index2_upper,
            index3_upper=index3_upper,
            index4_low=index4_low,
        )
        for index in index_set
    }


def _valid_scores(scores: dict[int, _PocketScore]) -> list[_PocketScore]:
    return [scores[idx] for idx in sorted(scores.keys()) if scores[idx].in_pocket]


def _choose_index_from_scores(scores: dict[int, _PocketScore]) -> _PocketScore | None:
    valid = _valid_scores(scores)
    if not valid:
        return None
    return valid[0]


@dataclass(frozen=True)
class _SemidirectGroup:
    order: int
    half: int
    twist: int
    kind: str

    def mul(self, g: int, h: int) -> int:
        a1, b1 = g % self.half, g // self.half
        a2, b2 = h % self.half, h // self.half
        acted = a2 if b1 == 0 else (self.twist * a2) % self.half
        a = (a1 + acted) % self.half
        b = b1 ^ b2
        return a + b * self.half

    def inv(self, g: int) -> int:
        a, b = g % self.half, g // self.half
        if b == 0:
            return (-a) % self.half
        inv_a = (-(self.twist * a)) % self.half
        return inv_a + self.half


@dataclass(frozen=True)
class _CyclicGroup:
    order: int
    kind: str = "cyclic"

    def mul(self, g: int, h: int) -> int:
        return (g + h) % self.order

    def inv(self, g: int) -> int:
        return (-g) % self.order


def _right_actions(group, generators: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    generators = np.asarray(generators, dtype=np.int32)[:, None]
    nodes = np.asarray(nodes, dtype=np.int32)[None, :]
    if isinstance(group, _CyclicGroup):
        return (nodes + generators) % group.order
    half = group.half
    acted = np.where(nodes // half == 0, generators % half, group.twist * (generators % half))
    return (nodes % half + acted) % half + ((nodes // half) ^ (generators // half)) * half


@lru_cache(maxsize=8)
def _adjacency_generator_table(group) -> np.ndarray:
    nodes = np.arange(group.order, dtype=np.int32)
    inverses = np.asarray([group.inv(int(g)) for g in nodes], dtype=np.int32)
    # Entry (g,h) is g^{-1}h: adjacency is the generator indicator at this index.
    table = _right_actions(group, nodes, inverses).T.copy()
    table.setflags(write=False)
    return table


def _build_group(target_n: int, index: int, lift_nodes: int, index2_group: str = "current"):
    if index2_group not in {"current", "cyclic"}:
        raise ValueError(f"Unknown index-2 group choice: {index2_group}")
    lifted_n = target_n + lift_nodes
    if index == 2:
        if lifted_n < 2 or lifted_n % 2 != 0:
            raise ValueError(f"Unsupported lifted order {lifted_n} for index-2.")
        if index2_group == "cyclic":
            return _CyclicGroup(order=lifted_n)
        half = lifted_n // 2
        use_semidihedral = (
            lift_nodes == 0
            and _is_power_of_two(target_n)
            and target_n >= 16
        )
        if use_semidihedral:
            k = int(math.log2(lifted_n))
            twist = (2 ** (k - 2) - 1) % half
            kind = "semidihedral"
        else:
            twist = (half - 1) % half if half > 1 else 0
            kind = "dihedral"
        return _SemidirectGroup(order=lifted_n, half=half, twist=twist, kind=kind)
    if index in (3, 4):
        return _CyclicGroup(order=lifted_n)
    raise ValueError(f"Unsupported index: {index}")


def _split_cosets(group, index: int) -> tuple[list[int], list[int]]:
    subgroup: list[int] = []
    nontrivial_union: list[int] = []

    for g in range(group.order):
        if index == 2 and isinstance(group, _SemidirectGroup):
            residue = 0 if g < group.half else 1
        else:
            residue = g % index
        if residue == 0:
            subgroup.append(g)
        else:
            nontrivial_union.append(g)

    return subgroup, nontrivial_union


class _InverseClosedSampler:
    def __init__(self, group, pool: list[int]) -> None:
        self.group = group
        self.pool = list(pool)
        self.pool_set = set(pool)
        self.involutions: list[int] = []
        self.pairs: list[tuple[int, int]] = []
        self._feasible_counts: dict[int, tuple[int, ...]] = {}
        self._build_parts()

    def _build_parts(self) -> None:
        visited: set[int] = set()
        for g in self.pool:
            if g in visited:
                continue
            h = self.group.inv(g)
            if h == g:
                self.involutions.append(g)
                visited.add(g)
                continue
            if h not in self.pool_set:
                continue
            visited.add(g)
            visited.add(h)
            self.pairs.append((g, h))

    def feasible(self, size: int) -> bool:
        return len(self._feasible_involution_counts(size)) > 0

    def sample(self, size: int, rng: random.Random) -> set[int]:
        if size < 0 or size > len(self.pool):
            raise ValueError(f"Invalid subset size={size}, pool={len(self.pool)}.")
        if size == 0:
            return set()

        feasible_t = self._feasible_involution_counts(size)
        if not feasible_t:
            raise ValueError(f"No inverse-closed subset of size {size}.")

        t = rng.choice(feasible_t)
        p = (size - t) // 2
        chosen_inv = rng.sample(self.involutions, t)
        chosen_pairs = rng.sample(self.pairs, p)

        out = set(chosen_inv)
        for a, b in chosen_pairs:
            out.add(a)
            out.add(b)
        return out

    def _feasible_involution_counts(self, size: int) -> tuple[int, ...]:
        if size in self._feasible_counts:
            return self._feasible_counts[size]
        min_t = max(0, size - 2 * len(self.pairs))
        max_t = min(size, len(self.involutions))
        out: list[int] = []
        for t in range(min_t, max_t + 1):
            rem = size - t
            if rem % 2 != 0:
                continue
            if rem // 2 <= len(self.pairs):
                out.append(t)
        result = tuple(out)
        self._feasible_counts[size] = result
        return result


def _sample_generators(
    *,
    nontrivial_union: list[int],
    subgroup_no_identity: list[int],
    desired_degree: int,
    rng: random.Random,
    nontrivial_sampler: _InverseClosedSampler,
    subgroup_sampler: _InverseClosedSampler,
) -> set[int] | None:
    if desired_degree <= 0:
        return None

    nontrivial_size = len(nontrivial_union)
    if desired_degree <= nontrivial_size:
        fixed: set[int] = set()
        sampler = nontrivial_sampler
        need = desired_degree
    else:
        fixed = set(nontrivial_union)
        sampler = subgroup_sampler
        need = desired_degree - nontrivial_size

    if not sampler.feasible(need):
        return None
    sampled = sampler.sample(need, rng)
    output = set(fixed)
    output.update(sampled)
    return output


def _sample_generators_random(
    *,
    desired_degree: int,
    rng: random.Random,
    full_sampler: _InverseClosedSampler,
) -> set[int] | None:
    if desired_degree <= 0:
        return None
    if not full_sampler.feasible(desired_degree):
        return None
    return full_sampler.sample(desired_degree, rng)


def _degree_sampler_need(
    *,
    desired_degree: int,
    nontrivial_size: int,
):
    if desired_degree <= 0:
        return None
    if desired_degree <= nontrivial_size:
        return "nontrivial", desired_degree
    return "subgroup", desired_degree - nontrivial_size


def _is_degree_feasible(
    *,
    desired_degree: int,
    nontrivial_size: int,
    nontrivial_sampler: _InverseClosedSampler,
    subgroup_sampler: _InverseClosedSampler,
) -> bool:
    sampler_need = _degree_sampler_need(
        desired_degree=desired_degree,
        nontrivial_size=nontrivial_size,
    )
    if sampler_need is None:
        return False
    which, need = sampler_need
    sampler = nontrivial_sampler if which == "nontrivial" else subgroup_sampler
    return sampler.feasible(need)


def _largest_feasible_degree_leq(
    *,
    max_degree: int,
    min_degree: int,
    nontrivial_size: int,
    nontrivial_sampler: _InverseClosedSampler,
    subgroup_sampler: _InverseClosedSampler,
) -> int | None:
    for degree in range(max_degree, min_degree - 1, -1):
        if _is_degree_feasible(
            desired_degree=degree,
            nontrivial_size=nontrivial_size,
            nontrivial_sampler=nontrivial_sampler,
            subgroup_sampler=subgroup_sampler,
        ):
            return degree
    return None


def _is_degree_feasible_random(
    *,
    desired_degree: int,
    full_sampler: _InverseClosedSampler,
) -> bool:
    if desired_degree <= 0:
        return False
    return full_sampler.feasible(desired_degree)


def _largest_feasible_degree_leq_random(
    *,
    max_degree: int,
    min_degree: int,
    full_sampler: _InverseClosedSampler,
) -> int | None:
    for degree in range(max_degree, min_degree - 1, -1):
        if _is_degree_feasible_random(
            desired_degree=degree,
            full_sampler=full_sampler,
        ):
            return degree
    return None


class _LiftDeleteBatchEvaluator:
    def __init__(
        self,
        *,
        group,
        delete_nodes: tuple[int, ...],
        backend: str,
        solver: str = "full",
    ) -> None:
        self.group = group
        self.n = group.order
        raw_delete = sorted(set(int(node) for node in delete_nodes))
        if any(node < 0 or node >= self.n for node in raw_delete):
            raise ValueError(
                f"delete_nodes must be within [0, {self.n - 1}], got {raw_delete}"
            )
        delete_arr = np.asarray(raw_delete, dtype=np.int32)
        keep_mask = np.ones((self.n,), dtype=bool)
        keep_mask[delete_arr] = False
        self.keep = np.nonzero(keep_mask)[0].astype(np.int32)
        if self.keep.size < 2:
            raise ValueError("Need at least 2 kept nodes to evaluate lambda_2.")

        if backend not in {"cpu", "cuda"}:
            raise ValueError(f"Unknown Cayley spectral backend: {backend}")
        if backend == "cuda" and cp is None:
            raise RuntimeError(
                "cayley_spectral_backend='cuda' requires CuPy, but it is unavailable."
            )
        self.backend = backend
        if solver not in {"full", "subset"}:
            raise ValueError(f"Unknown dense solver: {solver}")
        if backend == "cuda" and solver != "full":
            raise ValueError("Selected-eigenvalue solving is CPU-only.")
        self.solver = solver

        self.rows_np = np.arange(self.n, dtype=np.int32)
        self.diag_full_np = np.arange(self.n, dtype=np.int32)
        self.diag_keep_np = np.arange(self.keep.size, dtype=np.int32)
        self.adjacency_indices = _adjacency_generator_table(group)[np.ix_(self.keep, self.keep)]

        if self.backend == "cuda":  # pragma: no cover - optional CUDA path.
            self.right_action_np = _right_actions(group, self.rows_np, self.rows_np)
            self.rows_cp = cp.asarray(self.rows_np)
            self.diag_full_cp = cp.asarray(self.diag_full_np)
            self.diag_keep_cp = cp.asarray(self.diag_keep_np)
            self.keep_cp = cp.asarray(self.keep)
            self.right_action_cp = cp.asarray(self.right_action_np)

    def lambda2_edges_from_index_matrix(
        self,
        generator_index_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(generator_index_matrix, dtype=np.int32)
        if idx.ndim != 2:
            raise ValueError("generator_index_matrix must have shape [batch, degree].")
        if idx.shape[0] == 0:
            empty = np.empty((0,), dtype=np.float64)
            empty_i = np.empty((0,), dtype=np.int64)
            return empty, empty_i
        if idx.shape[1] == 0:
            zeros = np.zeros((idx.shape[0],), dtype=np.float64)
            zeros_i = np.zeros((idx.shape[0],), dtype=np.int64)
            return zeros, zeros_i
        if self.backend == "cuda":  # pragma: no cover - optional CUDA path.
            return self._lambda2_edges_cuda_batch(idx)
        return self._lambda2_edges_cpu_batch(idx)

    def _lambda2_edges_cpu_batch(
        self,
        generator_index_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(generator_index_matrix, dtype=np.int32)
        b, d = idx.shape
        n = self.n

        indicator = np.zeros((b, n), dtype=np.float64)
        indicator[np.arange(b)[:, None], idx] = 1.0
        indicator[:, 0] = 0.0
        sub = indicator[:, self.adjacency_indices]
        sub = np.maximum(sub, np.transpose(sub, (0, 2, 1)))
        deg = np.sum(sub, axis=2)
        edges = np.rint(np.sum(deg, axis=1) / 2.0).astype(np.int64)

        L = -sub
        L[:, self.diag_keep_np, self.diag_keep_np] = deg
        if self.solver == "subset":
            l2 = np.asarray([
                selected_eigvalsh(matrix, subset_by_index=(1, 1), driver="evr", check_finite=False)[0]
                for matrix in L
            ], dtype=np.float64)
        else:
            l2 = np.asarray(np.linalg.eigvalsh(L)[:, 1], dtype=np.float64)
        return l2, edges

    def _lambda2_edges_cuda_batch(  # pragma: no cover - optional CUDA path.
        self,
        generator_index_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        idx = cp.asarray(generator_index_matrix, dtype=cp.int32)
        b, d = idx.shape
        n = self.n

        A = cp.zeros((b, n, n), dtype=cp.float64)
        batch_idx = cp.arange(b, dtype=cp.int32)[:, None]
        rows = self.rows_cp[None, :]
        for j in range(d):
            s = idx[:, j]
            cols = self.right_action_cp[s]
            A[batch_idx, rows, cols] = 1.0
        A = cp.maximum(A, cp.transpose(A, (0, 2, 1)))
        A[:, self.diag_full_cp, self.diag_full_cp] = 0.0

        sub = A[:, self.keep_cp][:, :, self.keep_cp]
        deg = cp.sum(sub, axis=2)
        edges = cp.rint(cp.sum(deg, axis=1) / 2.0).astype(cp.int64)

        L = -sub
        L[:, self.diag_keep_cp, self.diag_keep_cp] = deg
        evals = cp.linalg.eigvalsh(L)
        l2 = cp.asnumpy(evals[:, 1])
        edges_np = cp.asnumpy(edges)
        return np.asarray(l2, dtype=np.float64), np.asarray(edges_np, dtype=np.int64)


class _CyclicNoDeleteCharacteristicEvaluator:
    r"""
    Exact Laplacian lambda_2 for cyclic Cayley graphs with no node deletion.

    For G = Z_n and generator indicator a \in {0,1}^n, adjacency eigenvalues are
    the DFT coefficients of a. Since L = kI - A (k-regular), lambda_2 is:
        lambda_2 = k - max_{j != 0} lambda_j(A)
    """

    def __init__(self, *, group: _CyclicGroup) -> None:
        self.group = group
        self.n = int(group.order)
        if self.n < 2:
            raise ValueError("Need n >= 2 for cyclic characteristic evaluation.")

    def lambda2_edges_from_index_matrix(
        self,
        generator_index_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(generator_index_matrix, dtype=np.int32)
        if idx.ndim != 2:
            raise ValueError("generator_index_matrix must have shape [batch, degree].")
        if idx.shape[0] == 0:
            empty = np.empty((0,), dtype=np.float64)
            empty_i = np.empty((0,), dtype=np.int64)
            return empty, empty_i
        if idx.shape[1] == 0:
            zeros = np.zeros((idx.shape[0],), dtype=np.float64)
            zeros_i = np.zeros((idx.shape[0],), dtype=np.int64)
            return zeros, zeros_i

        b, d = idx.shape
        n = self.n

        # Indicator of generator set for each sample row.
        indicator = np.zeros((b, n), dtype=np.float64)
        row_idx = np.arange(b, dtype=np.int32)[:, None]
        indicator[row_idx, idx] = 1.0
        indicator[:, 0] = 0.0  # identity is never a valid generator

        # Adjacency eigenvalues are exact DFT coefficients for circulant Cayley adjacency.
        adj_eigs = np.fft.fft(indicator, axis=1).real
        max_nontrivial_adj = np.max(adj_eigs[:, 1:], axis=1)
        degree = np.sum(indicator, axis=1)

        lambda2 = np.maximum(degree - max_nontrivial_adj, 0.0)
        edges = np.rint(degree * float(n) / 2.0).astype(np.int64)
        return np.asarray(lambda2, dtype=np.float64), np.asarray(edges, dtype=np.int64)


class _SemidihedralNoDeleteCharacteristicEvaluator:
    """Exact lambda_2 of inverse-closed semidihedral Cayley graphs, without deletion.

    Write n = 2m and t = m/2 - 1. Fourier transformation along the normal
    cyclic subgroup gives m Hermitian blocks [[R_j, T_j], [conj(T_j), R_tj]],
    where R and T are the FFTs of the two generator-coset indicators.
    Evaluating their eigenvalues costs O(n log n) time and O(n) space per graph.
    """

    def __init__(self, *, group: _SemidirectGroup) -> None:
        self.group = group
        self.n = int(group.order)
        self.m = int(group.half)
        if (
            self.n < 16
            or not _is_power_of_two(self.n)
            or self.n != 2 * self.m
            or int(group.twist) % self.m != self.m // 2 - 1
        ):
            raise ValueError("Expected a semidihedral group of power-of-two order >= 16.")
        self._twisted_frequencies = (np.arange(self.m) * int(group.twist)) % self.m
        self._inverse_indices = np.asarray([group.inv(g) for g in range(self.n)])

    def lambda2_edges_from_index_matrix(
        self,
        generator_index_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(generator_index_matrix, dtype=np.int32)
        if idx.ndim != 2:
            raise ValueError("generator_index_matrix must have shape [batch, degree].")
        b = idx.shape[0]
        if b == 0 or idx.shape[1] == 0:
            return np.zeros(b, dtype=np.float64), np.zeros(b, dtype=np.int64)
        if np.any(idx <= 0) or np.any(idx >= self.n):
            raise ValueError("Generators must be nonidentity group elements.")

        indicator = np.zeros((b, self.n), dtype=np.float64)
        indicator[np.arange(b)[:, None], idx] = 1.0
        if not np.array_equal(indicator, indicator[:, self._inverse_indices]):
            raise ValueError("The semidihedral generator sets must be inverse-closed.")

        rotations = np.fft.fft(indicator[:, :self.m], axis=1).real
        other_coset = np.fft.fft(indicator[:, self.m:], axis=1)
        twisted_rotations = rotations[:, self._twisted_frequencies]
        center = 0.5 * (rotations + twisted_rotations)
        radius = np.hypot(0.5 * (rotations - twisted_rotations), np.abs(other_coset))
        upper = center + radius

        # Remove only the trivial eigenvalue (the upper root at frequency zero).
        # Other copies of degree must remain, giving lambda_2 = 0 if disconnected.
        max_nontrivial_adj = np.maximum(
            center[:, 0] - radius[:, 0], np.max(upper[:, 1:], axis=1)
        )
        degree = np.sum(indicator, axis=1)
        lambda2 = np.maximum(degree - max_nontrivial_adj, 0.0)
        edges = np.rint(degree * float(self.n) / 2.0).astype(np.int64)
        return np.asarray(lambda2, dtype=np.float64), edges


class _DihedralNoDeleteCharacteristicEvaluator:
    """
    Exact Laplacian lambda_2 for dihedral Cayley graphs with no node deletion.

    Supports groups encoded by _SemidirectGroup with twist == -1 (mod half), i.e.,
    D_m of order 2m. Adjacency eigenvalues are computed from irreducible characters:
    1D irreps (2 or 4) plus all 2D irreps.
    """

    def __init__(self, *, group: _SemidirectGroup) -> None:
        if int(group.order) % 2 != 0:
            raise ValueError("Dihedral evaluator requires even group order.")
        if int(group.twist) % int(group.half) != (int(group.half) - 1) % int(group.half):
            raise ValueError("Dihedral evaluator requires twist == -1 mod half.")

        self.group = group
        self.n = int(group.order)
        self.m = int(group.half)
        if self.m < 1:
            raise ValueError("Invalid dihedral half-order.")

        # 2D irrep indices l:
        # - m odd:  l = 1..(m-1)/2
        # - m even: l = 1..(m/2 - 1)
        l_max = (self.m - 1) // 2 if (self.m % 2 == 1) else (self.m // 2 - 1)
        self._l_values = np.arange(1, max(0, l_max) + 1, dtype=np.int32)

        if self._l_values.size > 0:
            a = np.arange(self.m, dtype=np.float64)[None, :]
            l = self._l_values.astype(np.float64)[:, None]
            theta = 2.0 * np.pi * l * a / float(self.m)
            self._cos = np.cos(theta)
            self._sin = np.sin(theta)
        else:
            self._cos = np.empty((0, self.m), dtype=np.float64)
            self._sin = np.empty((0, self.m), dtype=np.float64)

        self._parity_a = np.where((np.arange(self.m, dtype=np.int32) % 2) == 0, 1.0, -1.0)

    def lambda2_edges_from_index_matrix(
        self,
        generator_index_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        idx = np.asarray(generator_index_matrix, dtype=np.int32)
        if idx.ndim != 2:
            raise ValueError("generator_index_matrix must have shape [batch, degree].")
        if idx.shape[0] == 0:
            empty = np.empty((0,), dtype=np.float64)
            empty_i = np.empty((0,), dtype=np.int64)
            return empty, empty_i
        if idx.shape[1] == 0:
            zeros = np.zeros((idx.shape[0],), dtype=np.float64)
            zeros_i = np.zeros((idx.shape[0],), dtype=np.int64)
            return zeros, zeros_i

        bsz, d = idx.shape
        m = self.m

        a = np.mod(idx, m).astype(np.int32)
        b = (idx // m).astype(np.int32)  # 0=rotation coset, 1=reflection coset
        rot_mask = (b == 0)
        refl_sign = np.where(rot_mask, 1.0, -1.0)
        degree = np.full((bsz,), float(d), dtype=np.float64)

        # Track largest adjacency eigenvalue among nontrivial irreps.
        max_nontrivial_adj = np.full((bsz,), -np.inf, dtype=np.float64)

        # 1D irrep: r -> +1, s -> -1.
        cnt_rot = np.sum(rot_mask, axis=1, dtype=np.int32)
        cnt_refl = d - cnt_rot
        chi2 = cnt_rot.astype(np.float64) - cnt_refl.astype(np.float64)
        max_nontrivial_adj = np.maximum(max_nontrivial_adj, chi2)

        # Additional two 1D irreps exist when m is even.
        if (m % 2) == 0:
            parity = self._parity_a[a]  # (-1)^a
            chi3 = np.sum(parity, axis=1)
            chi4 = np.sum(parity * refl_sign, axis=1)
            max_nontrivial_adj = np.maximum(max_nontrivial_adj, chi3)
            max_nontrivial_adj = np.maximum(max_nontrivial_adj, chi4)

        # 2D irreps: rho_l, l in self._l_values.
        for li in range(self._l_values.size):
            cos = self._cos[li][a]
            sin = self._sin[li][a]

            # M_l = sum_{(a,b) in S} R(a*theta_l) F^b
            # where F = diag(1,-1)
            a11 = np.sum(cos, axis=1)
            a12 = np.sum(np.where(rot_mask, -sin, sin), axis=1)
            a21 = np.sum(sin, axis=1)
            a22 = np.sum(np.where(rot_mask, cos, -cos), axis=1)

            tr = a11 + a22
            det = a11 * a22 - a12 * a21
            disc = np.sqrt(np.maximum(tr * tr - 4.0 * det, 0.0))
            e1 = 0.5 * (tr + disc)
            e2 = 0.5 * (tr - disc)
            pair_max = np.maximum(e1, e2)
            max_nontrivial_adj = np.maximum(max_nontrivial_adj, pair_max)

        lambda2 = np.maximum(degree - max_nontrivial_adj, 0.0)
        edges = np.rint(degree * float(self.n) / 2.0).astype(np.int64)
        return np.asarray(lambda2, dtype=np.float64), np.asarray(edges, dtype=np.int64)


def _build_batch_evaluator(
    *,
    group,
    delete_nodes: tuple[int, ...],
    config: DesignConfig,
) -> tuple[object, str]:
    eval_mode = str(getattr(config, "cayley_eval_mode", "dense")).strip().lower()
    solver = str(config.cayley_dense_solver).strip().lower()
    if solver not in {"full", "subset", "fft_iterative"}:
        raise ValueError(f"Unknown Cayley dense solver: {solver}")
    if solver != "full" and config.cayley_spectral_backend != "cpu":
        raise ValueError("Experimental Cayley solvers require the CPU backend.")
    if eval_mode not in {"dense", "character"}:
        raise ValueError(
            "cayley_eval_mode must be one of {'dense','character'}, "
            f"got {eval_mode!r}."
        )

    if (
        eval_mode == "character"
        and not delete_nodes
        and isinstance(group, _CyclicGroup)
    ):
        return (
            _CyclicNoDeleteCharacteristicEvaluator(group=group),
            "character_cyclic_no_delete",
        )

    if (
        eval_mode == "character"
        and not delete_nodes
        and isinstance(group, _SemidirectGroup)
        and group.kind == "semidihedral"
    ):
        return (
            _SemidihedralNoDeleteCharacteristicEvaluator(group=group),
            "character_semidihedral_no_delete",
        )

    if (
        eval_mode == "character"
        and not delete_nodes
        and isinstance(group, _SemidirectGroup)
        and int(group.twist) % int(group.half) == (int(group.half) - 1) % int(group.half)
    ):
        return (
            _DihedralNoDeleteCharacteristicEvaluator(group=group),
            "character_dihedral_no_delete",
        )

    if solver == "fft_iterative" and delete_nodes and isinstance(group, _CyclicGroup):
        return (
            CyclicDeleteFFTEvaluator(order=group.order, delete_nodes=delete_nodes),
            "fft_iterative_cyclic_delete",
        )

    return (
        _LiftDeleteBatchEvaluator(
            group=group,
            delete_nodes=delete_nodes,
            backend=config.cayley_spectral_backend,
            solver="full" if solver == "full" else "subset",
        ),
        "dense_laplacian" if solver == "full" else "dense_subset_laplacian",
    )


def _build_cayley_graph(group, generators: set[int]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(group.order))
    nodes = np.arange(group.order, dtype=np.int32)
    destinations = _right_actions(group, np.asarray(sorted(generators), dtype=np.int32), nodes)
    sources = np.broadcast_to(nodes, destinations.shape)
    inverse_closed = all(group.inv(g) in generators for g in generators)
    mask = sources < destinations if inverse_closed else sources != destinations
    graph.add_edges_from(zip(sources[mask].tolist(), destinations[mask].tolist()))
    return graph


@dataclass
class CayleyBackboneGenerator(BackboneGenerator):
    family: str = "cayley"

    def generate(
        self,
        problem: DesignProblem,
        config: DesignConfig,
        rng: random.Random,
    ) -> BackboneCandidate | None:
        eval_mode = str(getattr(config, "cayley_eval_mode", "dense")).strip().lower()
        if eval_mode not in {"dense", "character"}:
            raise ValueError(
                "cayley_eval_mode must be one of {'dense','character'}, "
                f"got {eval_mode!r}."
            )
        sampling_mode = str(config.cayley_sampling_mode).strip().lower()
        if sampling_mode not in {"coset", "random"}:
            raise ValueError(
                "cayley_sampling_mode must be one of {'coset','random'}, "
                f"got {sampling_mode!r}."
            )
        enable_index4 = bool(config.cayley_enable_index4)
        index2_upper = float(config.cayley_index2_overlap_high)
        index3_upper = float(config.cayley_index3_overlap_high)
        index4_low = float(config.cayley_index4_overlap_low)
        if not (0.0 < index2_upper < 0.75):
            raise ValueError(
                "cayley_index2_overlap_high must satisfy 0 < value < 0.75, "
                f"got {index2_upper!r}."
            )
        if not (0.5 < index3_upper < 0.75):
            raise ValueError(
                "cayley_index3_overlap_high must satisfy 0.5 < value < 0.75, "
                f"got {index3_upper!r}."
            )
        if enable_index4 and not (0.5 < index4_low < 0.75):
            raise ValueError(
                "cayley_index4_overlap_low must satisfy 0.5 < value < 0.75, "
                f"got {index4_low!r}."
            )
        scores = _score_index_candidates(
            n=problem.n,
            m=problem.m,
            index2_upper=index2_upper,
            index3_upper=index3_upper,
            index4_low=index4_low,
            enable_index4=enable_index4,
        )
        valid_scores = _valid_scores(scores)
        if not valid_scores:
            return None

        if not config.cayley_multi_index_overlap:
            selected = _choose_index_from_scores(scores)
            if selected is None:
                return None
            return self._generate_for_score(
                problem=problem,
                config=config,
                rng=rng,
                selected=selected,
                scores=scores,
                index2_upper=index2_upper,
                index3_upper=index3_upper,
                index4_low=index4_low,
                enable_index4=enable_index4,
                sampling_mode=sampling_mode,
                evaluated_indices=[selected.index],
            )

        candidates: list[BackboneCandidate] = []
        evaluated_indices: list[int] = [score.index for score in valid_scores]
        # Preallocate streams so changing one group's sampler cannot perturb another index.
        index_rngs = (
            {score.index: random.Random(rng.getrandbits(64)) for score in valid_scores}
            if config.cayley_independent_index_seeds else {}
        )
        for selected in valid_scores:
            candidate = self._generate_for_score(
                problem=problem,
                config=config,
                rng=index_rngs.get(selected.index, rng),
                selected=selected,
                scores=scores,
                index2_upper=index2_upper,
                index3_upper=index3_upper,
                index4_low=index4_low,
                enable_index4=enable_index4,
                sampling_mode=sampling_mode,
                evaluated_indices=evaluated_indices,
            )
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            return None
        return max(candidates, key=self._candidate_key)

    @staticmethod
    def _candidate_key(candidate: BackboneCandidate) -> tuple[float, int]:
        return (candidate.backbone_lambda2, candidate.graph.number_of_edges())

    def _generate_for_score(
        self,
        *,
        problem: DesignProblem,
        config: DesignConfig,
        rng: random.Random,
        selected: _PocketScore,
        scores: dict[int, _PocketScore],
        index2_upper: float,
        index3_upper: float,
        index4_low: float,
        enable_index4: bool,
        sampling_mode: str,
        evaluated_indices: list[int],
    ) -> BackboneCandidate | None:
        group = _build_group(
            target_n=problem.n,
            index=selected.index,
            lift_nodes=selected.lift_nodes,
            index2_group=config.cayley_index2_group,
        )
        subgroup, nontrivial_union = _split_cosets(group, selected.index)
        if not nontrivial_union:
            return None

        subgroup_no_identity = [g for g in subgroup if g != 0]
        nontrivial_sampler = _InverseClosedSampler(group, nontrivial_union)
        subgroup_sampler = _InverseClosedSampler(group, subgroup_no_identity)
        full_no_identity = [g for g in range(group.order) if g != 0]
        full_sampler = _InverseClosedSampler(group, full_no_identity)

        delete_nodes: tuple[int, ...] = ()
        if selected.lift_nodes > 0:
            delete_nodes = tuple(sorted(rng.sample(range(group.order), selected.lift_nodes)))

        max_degree = min(group.order - 1, selected.k_max)
        if sampling_mode == "random":
            feasible_cap = _largest_feasible_degree_leq_random(
                max_degree=max_degree,
                min_degree=1,
                full_sampler=full_sampler,
            )
        else:
            feasible_cap = _largest_feasible_degree_leq(
                max_degree=max_degree,
                min_degree=1,
                nontrivial_size=len(nontrivial_union),
                nontrivial_sampler=nontrivial_sampler,
                subgroup_sampler=subgroup_sampler,
            )
        if feasible_cap is None:
            return None

        low_d = max(1, feasible_cap - config.cayley_degree_slack)
        if sampling_mode == "random":
            feasible_degrees = [
                degree
                for degree in range(low_d, feasible_cap + 1)
                if _is_degree_feasible_random(
                    desired_degree=degree,
                    full_sampler=full_sampler,
                )
            ]
        else:
            feasible_degrees = [
                degree
                for degree in range(low_d, feasible_cap + 1)
                if _is_degree_feasible(
                    desired_degree=degree,
                    nontrivial_size=len(nontrivial_union),
                    nontrivial_sampler=nontrivial_sampler,
                    subgroup_sampler=subgroup_sampler,
                )
            ]
        if not feasible_degrees:
            return None

        sample_count = max(1, config.cayley_samples)
        samples_by_degree: dict[int, list[np.ndarray]] = defaultdict(list)
        seen_by_degree: dict[int, set[tuple[int, ...]]] = defaultdict(set)
        for _ in range(sample_count):
            desired_degree = (
                rng.choice(feasible_degrees)
                if len(feasible_degrees) > 1
                else feasible_degrees[0]
            )
            if sampling_mode == "random":
                generators = _sample_generators_random(
                    desired_degree=desired_degree,
                    rng=rng,
                    full_sampler=full_sampler,
                )
            else:
                generators = _sample_generators(
                    nontrivial_union=nontrivial_union,
                    subgroup_no_identity=subgroup_no_identity,
                    desired_degree=desired_degree,
                    rng=rng,
                    nontrivial_sampler=nontrivial_sampler,
                    subgroup_sampler=subgroup_sampler,
                )
            if not generators:
                continue
            key = tuple(sorted(generators))
            degree = len(key)
            if key in seen_by_degree[degree]:
                continue
            seen_by_degree[degree].add(key)
            samples_by_degree[degree].append(np.asarray(key, dtype=np.int32))

        if not samples_by_degree:
            return None

        evaluator, eval_backend = _build_batch_evaluator(
            group=group,
            delete_nodes=delete_nodes,
            config=config,
        )
        batch_size = max(1, config.cayley_eval_batch_size)

        best_key: tuple[float, int] | None = None
        best_indices: np.ndarray | None = None
        best_lambda2: float | None = None
        best_m_after: int | None = None
        best_m_lift: int | None = None
        best_deleted: int | None = None

        for degree, rows in samples_by_degree.items():
            for start in range(0, len(rows), batch_size):
                chunk = rows[start : start + batch_size]
                mat = np.stack(chunk, axis=0)
                l2_vals, edge_vals = evaluator.lambda2_edges_from_index_matrix(mat)
                m_lift = int((degree * group.order) // 2)
                for idx, generators_idx in enumerate(chunk):
                    l2 = float(l2_vals[idx])
                    m_after = int(edge_vals[idx])
                    deleted_edges = m_lift - m_after
                    if m_after > problem.m:
                        continue
                    if config.force_connected_backbone and l2 <= 1e-9:
                        continue
                    key = (l2, m_after)
                    if best_key is None or key > best_key:
                        best_key = key
                        best_indices = generators_idx
                        best_lambda2 = l2
                        best_m_after = m_after
                        best_m_lift = m_lift
                        best_deleted = deleted_edges

        if best_indices is None or best_lambda2 is None:
            return None

        graph = _build_cayley_graph(group, set(int(x) for x in best_indices))
        if delete_nodes:
            graph = graph.copy()
            graph.remove_nodes_from(delete_nodes)
            graph = relabel_to_integers(graph)

        if graph.number_of_nodes() != problem.n:
            return None
        if graph.number_of_edges() > problem.m:
            return None
        if config.force_connected_backbone and not nx.is_connected(graph):
            return None

        metadata: dict[str, float | int | bool | str] = {
            "group_kind": group.kind,
            "index2_group_choice": config.cayley_index2_group,
            "independent_index_seeds": bool(config.cayley_independent_index_seeds),
            "group_order": group.order,
            "selected_index": selected.index,
            "j_selected": selected.lift_nodes,
            "n_lift": selected.lifted_n,
            "k_max_selected": selected.k_max,
            "k_feasible_cap": int(feasible_cap),
            "m_proj_selected": selected.m_proj,
            "e_selected": selected.edge_ratio,
            "generator_size": int(best_indices.size),
            "m_lift": int(best_m_lift if best_m_lift is not None else 0),
            "m_after_delete": int(best_m_after if best_m_after is not None else 0),
            "deleted_edges": int(best_deleted if best_deleted is not None else 0),
            "sample_count_requested": int(sample_count),
            "sample_count_unique": int(sum(len(rows) for rows in samples_by_degree.values())),
            "delete_nodes_count": int(len(delete_nodes)),
            "delete_nodes": ",".join(str(node) for node in delete_nodes),
            "eval_mode": str(getattr(config, "cayley_eval_mode", "dense")),
            "eval_backend": str(eval_backend),
            "dense_solver": str(config.cayley_dense_solver),
            "iterative_fallback_count": int(getattr(evaluator, "fallback_count", 0)),
            "iterative_matvec_count": int(getattr(evaluator, "matvec_count", 0)),
            "multi_index_overlap": bool(config.cayley_multi_index_overlap),
            "sampling_mode": str(sampling_mode),
            "enable_index4": bool(enable_index4),
            "index2_pocket_high": float(index2_upper),
            "index3_pocket_high": float(index3_upper),
            "index4_pocket_low": float(index4_low),
            "evaluated_indices": ",".join(str(x) for x in evaluated_indices),
        }
        for index, score in scores.items():
            metadata[f"j_{index}"] = score.lift_nodes
            metadata[f"k_max_{index}"] = score.k_max
            metadata[f"m_proj_{index}"] = score.m_proj
            metadata[f"e_{index}"] = score.edge_ratio
            metadata[f"in_pocket_{index}"] = score.in_pocket

        return BackboneCandidate(
            family=self.family,
            graph=graph,
            backbone_lambda2=best_lambda2,
            metadata=metadata,
        )
