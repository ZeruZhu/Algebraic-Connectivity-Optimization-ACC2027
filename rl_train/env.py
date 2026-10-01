from __future__ import annotations

import random
from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .backbone_init import default_path_init_metadata, normalize_init_metadata
from .curriculum import CurriculumScheduler
from .features.structural import (
    build_edge_index,
    build_global_features as build_global_features,
    build_node_features as build_node_features,
)
from .features.candidates import (
    build_candidate_features as build_candidate_features,
)
from .graph_math import (
    algebraic_connectivity,
    build_path_adjacency,
    edge_count,
    m_target_from_rho,
    normalized_density,
    spectral_features,
    truncated_all_pairs_shortest_path,
)

InitAdjBuilder = Callable[[int, float, int], Tuple[np.ndarray, Dict[str, Any]]]


@dataclass
class GraphObservation:
    node_features: np.ndarray
    edge_index: np.ndarray
    candidate_pairs: np.ndarray
    pair_features: np.ndarray
    global_features: np.ndarray
    lambda2_norm: float
    n: int
    rho_target: float
    rho_current: float
    candidate_pool_size_raw: int = 0
    candidate_pool_size_selected: int = 0
    time_obs_build_sec: float = 0.0
    time_candidate_sec: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "node_features": self.node_features,
            "edge_index": self.edge_index,
            "candidate_pairs": self.candidate_pairs,
            "pair_features": self.pair_features,
            "global_features": self.global_features,
            "lambda2_norm": self.lambda2_norm,
            "n": self.n,
            "rho_target": self.rho_target,
            "rho_current": self.rho_current,
            "candidate_pool_size_raw": self.candidate_pool_size_raw,
            "candidate_pool_size_selected": self.candidate_pool_size_selected,
            "time_obs_build_sec": self.time_obs_build_sec,
            "time_candidate_sec": self.time_candidate_sec,
        }

    @staticmethod
    def from_dict(data: Dict) -> "GraphObservation":
        return GraphObservation(
            node_features=data["node_features"],
            edge_index=data["edge_index"],
            candidate_pairs=data["candidate_pairs"],
            pair_features=data["pair_features"],
            global_features=data["global_features"],
            lambda2_norm=float(data["lambda2_norm"]),
            n=int(data["n"]),
            rho_target=float(data["rho_target"]),
            rho_current=float(data["rho_current"]),
            candidate_pool_size_raw=int(data.get("candidate_pool_size_raw", 0)),
            candidate_pool_size_selected=int(data.get("candidate_pool_size_selected", 0)),
            time_obs_build_sec=float(data.get("time_obs_build_sec", 0.0)),
            time_candidate_sec=float(data.get("time_candidate_sec", 0.0)),
        )


class GraphEnv:
    def __init__(
        self,
        env_id: int,
        scheduler: CurriculumScheduler,
        top_k: int,
        dist_cap: int,
        terminal_bonus_coef: float,
        seed: int,
        rl_variant: str = "lite_v3",
        compute_spectral_each_step: bool = True,
        incremental_observation: bool = True,
        lite_v3_node_budget: int = 32,
        lite_v3_pair_budget: int = 96,
        lite_v3_pair_budget_min: int = 32,
        inference_only: bool = False,
    ):
        self.env_id = env_id
        self.scheduler = scheduler
        self.top_k = top_k
        self.dist_cap = dist_cap
        self.terminal_bonus_coef = terminal_bonus_coef
        self.rl_variant = str(rl_variant)
        self.compute_spectral_each_step = bool(compute_spectral_each_step)
        self.incremental_observation = bool(incremental_observation)
        self.inference_only = bool(inference_only)
        if self.inference_only and (self.rl_variant != "lite_v3" or self.compute_spectral_each_step):
            raise ValueError("inference_only requires lite_v3 with per-step spectral evaluation disabled")
        if self.rl_variant not in {"lite_v3"}:
            raise ValueError(f"Unsupported rl_variant: {self.rl_variant}")
        self.lite_v3_node_budget = int(lite_v3_node_budget)
        self.lite_v3_pair_budget = int(lite_v3_pair_budget)
        self.lite_v3_pair_budget_min = int(lite_v3_pair_budget_min)

        self.py_rng = random.Random(seed + 10007 * env_id)
        self.np_rng = np.random.RandomState(seed + 20011 * env_id)
        self._init_seed_base = int(seed + 30013 * env_id)

        self.n: int = 0
        self.adj: Optional[np.ndarray] = None
        self.rho_target: float = 0.0
        self.m_target: int = 0
        self.episode_return: float = 0.0
        self.episode_len: int = 0
        self.current_lambda2: float = 0.0
        self.current_lambda3: float = 0.0
        self.current_phi2: Optional[np.ndarray] = None
        self.current_phi3: Optional[np.ndarray] = None
        self._deg_cache: Optional[np.ndarray] = None
        self._a2_counts: Optional[np.ndarray] = None
        self._triangles_per_node: Optional[np.ndarray] = None
        self._dist_matrix: Optional[np.ndarray] = None
        self._pair_i: Optional[np.ndarray] = None
        self._pair_j: Optional[np.ndarray] = None
        self._pair_is_nonedge: Optional[np.ndarray] = None
        self._pair_to_index: Optional[np.ndarray] = None
        self._eye_mask: Optional[np.ndarray] = None
        self._episode_init_metadata: Dict[str, Any] = default_path_init_metadata(0)

    def _compute_all_pairs_shortest_path(self, adj: np.ndarray) -> np.ndarray:
        return truncated_all_pairs_shortest_path(adj, len(adj) + 1)

    def _initialize_incremental_state(self) -> None:
        if self.adj is None or (self.inference_only and edge_count(self.adj) >= self.m_target):
            self._deg_cache = None
            self._a2_counts = None
            self._triangles_per_node = None
            self._dist_matrix = None
            self._pair_i = None
            self._pair_j = None
            self._pair_is_nonedge = None
            self._pair_to_index = None
            self._eye_mask = None
            return

        n = self.adj.shape[0]
        adj_bool = self.adj > 0
        adj_f = adj_bool.astype(np.float64)

        self._deg_cache = np.sum(adj_bool, axis=1, dtype=np.float64)
        # Binary dot products are integer-exact here and use the float BLAS kernel.
        self._a2_counts = (adj_f @ adj_f).astype(np.int64)
        self._triangles_per_node = np.sum(self._a2_counts * adj_bool, axis=1, dtype=np.float64) / 2.0
        self._dist_matrix = self._compute_all_pairs_shortest_path(self.adj)

        rows, cols = np.triu_indices(n, k=1)
        self._pair_i = rows.astype(np.int64, copy=False)
        self._pair_j = cols.astype(np.int64, copy=False)
        self._pair_is_nonedge = (self.adj[self._pair_i, self._pair_j] == 0)

        pair_to_index = np.full((n, n), -1, dtype=np.int64)
        pair_idx = np.arange(self._pair_i.size, dtype=np.int64)
        pair_to_index[self._pair_i, self._pair_j] = pair_idx
        pair_to_index[self._pair_j, self._pair_i] = pair_idx
        self._pair_to_index = pair_to_index
        self._eye_mask = np.eye(n, dtype=bool)

    def _update_incremental_state_after_add(
        self,
        i: int,
        j: int,
        adj_row_i_before: np.ndarray,
        adj_row_j_before: np.ndarray,
        common_neighbors: np.ndarray,
    ) -> None:
        if (
            self._deg_cache is None
            or self._a2_counts is None
            or self._triangles_per_node is None
            or self._dist_matrix is None
            or self._pair_is_nonedge is None
            or self._pair_to_index is None
        ):
            self._initialize_incremental_state()
            return

        pair_idx = int(self._pair_to_index[i, j])
        if pair_idx >= 0:
            self._pair_is_nonedge[pair_idx] = False

        self._deg_cache[i] += 1.0
        self._deg_cache[j] += 1.0

        c = float(common_neighbors.size)
        if c > 0.0:
            self._triangles_per_node[i] += c
            self._triangles_per_node[j] += c
            self._triangles_per_node[common_neighbors] += 1.0

        # Exact rank-2 update for A^2 after adding edge (i, j).
        self._a2_counts[:, j] += adj_row_i_before
        self._a2_counts[:, i] += adj_row_j_before
        self._a2_counts[j, :] += adj_row_i_before
        self._a2_counts[i, :] += adj_row_j_before
        self._a2_counts[i, i] += 1
        self._a2_counts[j, j] += 1

        old_dist = self._dist_matrix
        via_ij = old_dist[:, i : i + 1] + 1 + old_dist[j : j + 1, :]
        via_ji = old_dist[:, j : j + 1] + 1 + old_dist[i : i + 1, :]
        self._dist_matrix = np.minimum(old_dist, np.minimum(via_ij, via_ji)).astype(
            np.int64, copy=False
        )

    def reset(
        self,
        global_env_step: int,
        *,
        init_mode: str = "path",
        initial_adj_builder: Optional[InitAdjBuilder] = None,
    ) -> GraphObservation:
        self.n = self.scheduler.sample_n(global_env_step, self.py_rng)
        self.rho_target = self.py_rng.random()
        self.m_target = m_target_from_rho(self.n, self.rho_target)
        # Avoid zero-decision episodes in training when rho maps to m=n-1.
        if self.n > 2 and self.m_target <= self.n - 1:
            self.m_target = self.n

        if init_mode == "backbone":
            if initial_adj_builder is None:
                raise ValueError("init_mode='backbone' requires initial_adj_builder.")
            init_seed = self._init_seed_base
            init_adj, init_meta = initial_adj_builder(
                n=int(self.n),
                rho_target=float(self.rho_target),
                seed=int(init_seed),
            )
            adj = np.asarray(init_adj, dtype=np.uint8)
            if adj.shape != (self.n, self.n):
                raise ValueError(
                    f"initial_adj shape mismatch: got {adj.shape}, expected {(self.n, self.n)}"
                )
            if not np.array_equal(adj, adj.T):
                raise ValueError("initial_adj must be symmetric for undirected graph.")
            if np.any(np.diag(adj) != 0):
                raise ValueError("initial_adj diagonal must be zero.")

            self.adj = adj.copy()
            self._episode_init_metadata = normalize_init_metadata(
                init_meta,
                n=self.n,
                init_mode="backbone",
            )
            if edge_count(self.adj) > self.m_target:
                raise RuntimeError(
                    "Initialization edge budget violation: "
                    f"m_init={edge_count(self.adj)} > m_target={self.m_target}"
                )
            self.episode_return = 0.0
            self.episode_len = 0
            if self.incremental_observation:
                self._initialize_incremental_state()
            return self._initial_observation()

        self.adj = build_path_adjacency(self.n)
        self._episode_init_metadata = normalize_init_metadata(
            default_path_init_metadata(self.n),
            n=self.n,
            init_mode="path",
        )
        self.episode_return = 0.0
        self.episode_len = 0
        if self.incremental_observation:
            self._initialize_incremental_state()
        return self._initial_observation()

    def _initial_observation(self) -> GraphObservation:
        if self.inference_only:
            self.current_lambda2 = algebraic_connectivity(self.adj)
            self.current_lambda3 = 0.0
            self.current_phi2 = self.current_phi3 = None
            if edge_count(self.adj) >= self.m_target:
                return self._terminal_observation()
            return self._build_observation()
        lambda2, lambda3, phi2, phi3 = spectral_features(self.adj)
        return self._build_observation(spectral_cache=(lambda2, lambda3, phi2, phi3))

    def _terminal_observation(self) -> GraphObservation:
        """Inference consumers only need the terminal score, not another action state."""
        rho = normalized_density(self.n, edge_count(self.adj))
        return GraphObservation(
            node_features=np.zeros((self.n, 3), dtype=np.float32),
            edge_index=np.zeros((2, 0), dtype=np.int64),
            candidate_pairs=np.zeros((0, 2), dtype=np.int64),
            pair_features=np.zeros((0, 5), dtype=np.float32),
            global_features=build_global_features(
                n=self.n, rho_target=self.rho_target,
                rho_current=rho,
            ),
            lambda2_norm=self.current_lambda2 / max(1, self.n), n=self.n,
            rho_target=self.rho_target, rho_current=rho,
        )

    def reset_with_target(
        self,
        n: int,
        rho_target: float,
        initial_adj: Optional[np.ndarray] = None,
        init_mode: str = "path",
        init_metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphObservation:
        self.n = int(n)
        self.rho_target = float(rho_target)
        self.m_target = m_target_from_rho(self.n, self.rho_target)
        if initial_adj is None:
            self.adj = build_path_adjacency(self.n)
            self._episode_init_metadata = normalize_init_metadata(
                init_metadata if init_metadata is not None else default_path_init_metadata(self.n),
                n=self.n,
                init_mode="path",
            )
        else:
            adj = np.asarray(initial_adj, dtype=np.uint8)
            if adj.shape != (self.n, self.n):
                raise ValueError(
                    f"initial_adj shape mismatch: got {adj.shape}, expected {(self.n, self.n)}"
                )
            if not np.array_equal(adj, adj.T):
                raise ValueError("initial_adj must be symmetric for undirected graph.")
            if np.any(np.diag(adj) != 0):
                raise ValueError("initial_adj diagonal must be zero.")
            self.adj = adj.copy()
            self._episode_init_metadata = normalize_init_metadata(
                init_metadata,
                n=self.n,
                init_mode=init_mode,
            )
            if edge_count(self.adj) > self.m_target:
                raise RuntimeError(
                    "Initialization edge budget violation: "
                    f"m_init={edge_count(self.adj)} > m_target={self.m_target}"
                )
        self.episode_return = 0.0
        self.episode_len = 0
        if self.incremental_observation:
            self._initialize_incremental_state()
        return self._initial_observation()

    def _build_observation(
        self,
        *,
        spectral_cache: Optional[Tuple[float, float, np.ndarray, np.ndarray]] = None,
    ) -> GraphObservation:
        obs_build_start = time.perf_counter()
        if self.adj is None:
            raise RuntimeError("Environment not initialized")

        n = self.n
        m = edge_count(self.adj)
        rho_current = normalized_density(n, m)

        use_spectral = self.compute_spectral_each_step
        if spectral_cache is not None:
            lambda2, lambda3, phi2, phi3 = spectral_cache
            self.current_lambda2 = float(lambda2)
            self.current_lambda3 = float(lambda3)
            self.current_phi2 = np.asarray(phi2, dtype=np.float64).copy()
            self.current_phi3 = np.asarray(phi3, dtype=np.float64).copy()
        elif use_spectral:
            lambda2, lambda3, phi2, phi3 = spectral_features(self.adj)
            self.current_lambda2 = float(lambda2)
            self.current_lambda3 = float(lambda3)
            self.current_phi2 = np.asarray(phi2, dtype=np.float64).copy()
            self.current_phi3 = np.asarray(phi3, dtype=np.float64).copy()
        else:
            lambda2 = float(self.current_lambda2)
            lambda3 = float(self.current_lambda3)
            phi2 = (
                np.asarray(self.current_phi2, dtype=np.float64)
                if self.current_phi2 is not None
                else np.zeros((n,), dtype=np.float64)
            )
            phi3 = (
                np.asarray(self.current_phi3, dtype=np.float64)
                if self.current_phi3 is not None
                else np.zeros((n,), dtype=np.float64)
            )

        node_features, deg = build_node_features(
            adj=self.adj,
            n=n,
            incremental_observation=self.incremental_observation,
            deg_cache=self._deg_cache,
            a2_counts=self._a2_counts,
            triangles_per_node=self._triangles_per_node,
            eye_mask=self._eye_mask,
        )
        global_features = build_global_features(
            n=n,
            rho_target=self.rho_target,
            rho_current=rho_current,
        )
        cand_start = time.perf_counter()
        (
            candidate_pairs,
            pair_features,
            candidate_pool_size_raw,
            candidate_pool_size_selected,
        ) = build_candidate_features(
            adj=self.adj,
            deg=deg,
            node_features=node_features,
            n=n,
            dist_cap=self.dist_cap,
            incremental_observation=self.incremental_observation,
            pair_i=self._pair_i,
            pair_j=self._pair_j,
            pair_is_nonedge=self._pair_is_nonedge,
            dist_matrix=self._dist_matrix,
            node_budget=self.lite_v3_node_budget,
            pair_budget=self.lite_v3_pair_budget,
            pair_budget_min=self.lite_v3_pair_budget_min,
            a2_counts=self._a2_counts,
        )
        time_candidate_sec = float(time.perf_counter() - cand_start)
        edge_index = np.zeros((2, 0), dtype=np.int64) if self.inference_only else build_edge_index(self.adj)
        time_obs_build_sec = float(time.perf_counter() - obs_build_start)

        return GraphObservation(
            node_features=node_features,
            edge_index=edge_index,
            candidate_pairs=candidate_pairs,
            pair_features=pair_features,
            global_features=global_features,
            lambda2_norm=float(lambda2 / max(1, n)),
            n=n,
            rho_target=self.rho_target,
            rho_current=rho_current,
            candidate_pool_size_raw=int(candidate_pool_size_raw),
            candidate_pool_size_selected=int(candidate_pool_size_selected),
            time_obs_build_sec=time_obs_build_sec,
            time_candidate_sec=time_candidate_sec,
        )

    def step(
        self,
        action_pair: Tuple[int, int],
        extra_reward: float = 0.0,
    ) -> Tuple[GraphObservation, float, bool, Dict]:
        step_start = time.perf_counter()
        if self.adj is None:
            raise RuntimeError("Environment not initialized")

        i, j = action_pair
        if i == j or i < 0 or j < 0 or i >= self.n or j >= self.n:
            raise ValueError(f"Invalid action pair {action_pair}")
        if self.adj[i, j] == 1:
            raise ValueError(f"Action pair already an edge: {action_pair}")

        old_lambda2 = self.current_lambda2
        adj_row_i_before = self.adj[i].astype(np.int64, copy=True)
        adj_row_j_before = self.adj[j].astype(np.int64, copy=True)
        common_neighbors = np.flatnonzero(
            np.logical_and(adj_row_i_before > 0, adj_row_j_before > 0)
        ).astype(np.int64, copy=False)

        self.adj[i, j] = 1
        self.adj[j, i] = 1
        self.episode_len += 1
        if self.incremental_observation:
            self._update_incremental_state_after_add(
                i=i,
                j=j,
                adj_row_i_before=adj_row_i_before,
                adj_row_j_before=adj_row_j_before,
                common_neighbors=common_neighbors,
            )

        use_spectral = self.compute_spectral_each_step
        done = edge_count(self.adj) >= self.m_target
        reward = 0.0

        if use_spectral:
            new_lambda2, new_lambda3, phi2, phi3 = spectral_features(self.adj)
            self.current_lambda2 = float(new_lambda2)
            self.current_lambda3 = float(new_lambda3)
            self.current_phi2 = np.asarray(phi2, dtype=np.float64).copy()
            self.current_phi3 = np.asarray(phi3, dtype=np.float64).copy()
            reward = (new_lambda2 - old_lambda2) / float(max(1, self.n))
            if done:
                reward += self.terminal_bonus_coef * (new_lambda2 / float(max(1, self.n)))
            obs = self._build_observation(
                spectral_cache=(new_lambda2, new_lambda3, phi2, phi3)
            )
            terminal_lambda2_norm = (new_lambda2 / float(max(1, self.n))) if done else None
        else:
            # Fast inference: avoid per-step spectral decomposition.
            if done:
                if self.inference_only:
                    new_lambda2 = algebraic_connectivity(self.adj)
                    self.current_lambda2 = new_lambda2
                else:
                    new_lambda2, new_lambda3, phi2, phi3 = spectral_features(self.adj)
                    self.current_lambda2 = float(new_lambda2)
                    self.current_lambda3 = float(new_lambda3)
                    self.current_phi2 = np.asarray(phi2, dtype=np.float64).copy()
                    self.current_phi3 = np.asarray(phi3, dtype=np.float64).copy()
                terminal_lambda2_norm = new_lambda2 / float(max(1, self.n))
            else:
                terminal_lambda2_norm = None
            obs = self._terminal_observation() if done and self.inference_only else self._build_observation()

        reward += float(extra_reward)
        self.episode_return += reward

        info: Dict = {
            "episode_done": done,
            "episode_return": self.episode_return if done else None,
            "episode_len": self.episode_len if done else None,
            "terminal_lambda2_norm": terminal_lambda2_norm,
            "rho_target": self.rho_target,
            "n": self.n,
            "candidate_pool_size_raw": int(obs.candidate_pool_size_raw),
            "candidate_pool_size_selected": int(obs.candidate_pool_size_selected),
            "time_obs_build_sec": float(obs.time_obs_build_sec),
            "time_candidate_sec": float(obs.time_candidate_sec),
            "time_env_step_sec": float(time.perf_counter() - step_start),
        }
        if done:
            info.update(self._episode_init_metadata)
        return obs, float(reward), done, info

    def state_dict(self) -> Dict:
        return {
            "env_id": self.env_id,
            "rl_variant": self.rl_variant,
            "compute_spectral_each_step": self.compute_spectral_each_step,
            "incremental_observation": self.incremental_observation,
            "inference_only": self.inference_only,
            "lite_v3_node_budget": self.lite_v3_node_budget,
            "lite_v3_pair_budget": self.lite_v3_pair_budget,
            "lite_v3_pair_budget_min": self.lite_v3_pair_budget_min,
            "n": self.n,
            "adj": self.adj,
            "rho_target": self.rho_target,
            "m_target": self.m_target,
            "episode_return": self.episode_return,
            "episode_len": self.episode_len,
            "current_lambda2": self.current_lambda2,
            "current_lambda3": self.current_lambda3,
            "current_phi2": self.current_phi2,
            "current_phi3": self.current_phi3,
            "episode_init_metadata": dict(self._episode_init_metadata),
            "py_rng_state": self.py_rng.getstate(),
            "np_rng_state": self.np_rng.get_state(),
        }

    def load_state_dict(self, state: Dict) -> None:
        self.rl_variant = str(state.get("rl_variant", self.rl_variant))
        self.compute_spectral_each_step = bool(
            state.get("compute_spectral_each_step", self.compute_spectral_each_step)
        )
        self.inference_only = bool(state.get("inference_only", False))
        if self.inference_only and (self.rl_variant != "lite_v3" or self.compute_spectral_each_step):
            raise ValueError("Invalid inference_only environment state")
        self.lite_v3_node_budget = int(
            state.get("lite_v3_node_budget", self.lite_v3_node_budget)
        )
        self.lite_v3_pair_budget = int(
            state.get("lite_v3_pair_budget", self.lite_v3_pair_budget)
        )
        self.lite_v3_pair_budget_min = int(
            state.get("lite_v3_pair_budget_min", self.lite_v3_pair_budget_min)
        )
        self.n = int(state["n"])
        self.adj = state["adj"].copy() if state["adj"] is not None else None
        self.rho_target = float(state["rho_target"])
        self.m_target = int(state["m_target"])
        self.episode_return = float(state["episode_return"])
        self.episode_len = int(state["episode_len"])
        self.current_lambda2 = float(state["current_lambda2"])
        self.current_lambda3 = float(state["current_lambda3"])
        raw_phi2 = state.get("current_phi2")
        raw_phi3 = state.get("current_phi3")
        self.current_phi2 = (
            np.asarray(raw_phi2, dtype=np.float64).copy()
            if raw_phi2 is not None
            else None
        )
        self.current_phi3 = (
            np.asarray(raw_phi3, dtype=np.float64).copy()
            if raw_phi3 is not None
            else None
        )
        raw_meta = state.get("episode_init_metadata")
        raw_mode = "path"
        if isinstance(raw_meta, dict):
            raw_mode = str(raw_meta.get("init_mode", "path"))
        self._episode_init_metadata = normalize_init_metadata(
            raw_meta if isinstance(raw_meta, dict) else None,
            n=max(0, self.n),
            init_mode=raw_mode,
        )
        self.py_rng.setstate(state["py_rng_state"])
        self.np_rng.set_state(state["np_rng_state"])
        self.incremental_observation = bool(state.get("incremental_observation", True))
        if self.incremental_observation:
            self._initialize_incremental_state()


class VectorGraphEnvManager:
    def __init__(
        self,
        num_envs: int,
        scheduler: CurriculumScheduler,
        top_k: int,
        dist_cap: int,
        terminal_bonus_coef: float,
        base_seed: int,
        rl_variant: str = "lite_v3",
        compute_spectral_each_step: bool = True,
        incremental_observation: bool = True,
        lite_v3_node_budget: int = 32,
        lite_v3_pair_budget: int = 96,
        lite_v3_pair_budget_min: int = 32,
        init_mode: str = "path",
        initial_adj_builder: Optional[InitAdjBuilder] = None,
    ):
        if init_mode not in {"path", "backbone"}:
            raise ValueError(f"Unsupported init_mode: {init_mode}")
        if init_mode == "backbone" and initial_adj_builder is None:
            raise ValueError("init_mode='backbone' requires initial_adj_builder.")

        self.init_mode = init_mode
        self.rl_variant = str(rl_variant)
        self.compute_spectral_each_step = bool(compute_spectral_each_step)
        self.initial_adj_builder = initial_adj_builder
        self.envs: List[GraphEnv] = [
            GraphEnv(
                env_id=i,
                scheduler=scheduler,
                top_k=top_k,
                dist_cap=dist_cap,
                terminal_bonus_coef=terminal_bonus_coef,
                seed=base_seed,
                rl_variant=self.rl_variant,
                compute_spectral_each_step=self.compute_spectral_each_step,
                incremental_observation=incremental_observation,
                lite_v3_node_budget=lite_v3_node_budget,
                lite_v3_pair_budget=lite_v3_pair_budget,
                lite_v3_pair_budget_min=lite_v3_pair_budget_min,
            )
            for i in range(num_envs)
        ]
        self.current_obs: List[Optional[GraphObservation]] = [None for _ in range(num_envs)]

    def _reset_until_actionable(self, env: GraphEnv, global_env_step: int) -> GraphObservation:
        # In backbone mode it is possible to initialize directly at target edge budget
        # (m_init == m_target). Training should skip such zero-completion episodes.
        max_attempts = 256
        for _ in range(max_attempts):
            obs = env.reset(
                global_env_step,
                init_mode=self.init_mode,
                initial_adj_builder=self.initial_adj_builder,
            )
            if env.adj is None:
                raise RuntimeError("Environment adjacency missing after reset")
            if edge_count(env.adj) < env.m_target and obs.candidate_pairs.shape[0] > 0:
                return obs
        raise RuntimeError(
            "Unable to sample actionable episode after many attempts. "
            "This usually means initialization frequently saturates m_target."
        )

    def ensure_actionable_observations(self, global_env_step: int) -> None:
        for i, env in enumerate(self.envs):
            obs = self.current_obs[i]
            needs_reset = True
            if obs is not None and env.adj is not None:
                if edge_count(env.adj) < env.m_target and obs.candidate_pairs.shape[0] > 0:
                    needs_reset = False
            if needs_reset:
                self.current_obs[i] = self._reset_until_actionable(env, global_env_step)

    def reset_all(self, global_env_step: int) -> List[GraphObservation]:
        out = []
        for i, env in enumerate(self.envs):
            obs = self._reset_until_actionable(env, global_env_step)
            self.current_obs[i] = obs
            out.append(obs)
        return out

    def step(
        self,
        action_pairs: Sequence[Tuple[int, int]],
        global_env_step_after_batch: int,
        extra_rewards: Optional[Sequence[float]] = None,
    ) -> Tuple[List[GraphObservation], List[float], List[bool], List[Dict]]:
        next_obs: List[GraphObservation] = []
        rewards: List[float] = []
        dones: List[bool] = []
        infos: List[Dict] = []

        if extra_rewards is None:
            extra_rewards = [0.0 for _ in range(len(self.envs))]

        for i, env in enumerate(self.envs):
            obs, reward, done, info = env.step(action_pairs[i], extra_reward=float(extra_rewards[i]))
            if done:
                reset_obs = self._reset_until_actionable(env, global_env_step_after_batch)
                next_obs.append(reset_obs)
                self.current_obs[i] = reset_obs
            else:
                next_obs.append(obs)
                self.current_obs[i] = obs
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        return next_obs, rewards, dones, infos

    def state_dict(self) -> Dict:
        return {
            "init_mode": self.init_mode,
            "rl_variant": self.rl_variant,
            "compute_spectral_each_step": self.compute_spectral_each_step,
            "env_states": [env.state_dict() for env in self.envs],
            "current_obs": [obs.to_dict() if obs is not None else None for obs in self.current_obs],
        }

    def load_state_dict(self, state: Dict) -> None:
        checkpoint_init_mode = str(state.get("init_mode", self.init_mode))
        if checkpoint_init_mode != self.init_mode:
            raise ValueError(
                f"Mismatched init_mode in checkpoint: {checkpoint_init_mode} != {self.init_mode}"
            )
        checkpoint_variant = str(state.get("rl_variant", self.rl_variant))
        if checkpoint_variant != self.rl_variant:
            raise ValueError(
                f"Mismatched rl_variant in checkpoint: {checkpoint_variant} != {self.rl_variant}"
            )
        checkpoint_compute_spectral = bool(
            state.get("compute_spectral_each_step", self.compute_spectral_each_step)
        )
        if checkpoint_compute_spectral != self.compute_spectral_each_step:
            raise ValueError(
                "Mismatched compute_spectral_each_step in checkpoint: "
                f"{checkpoint_compute_spectral} != {self.compute_spectral_each_step}"
            )
        env_states = state["env_states"]
        if len(env_states) != len(self.envs):
            raise ValueError("Mismatched number of environments in checkpoint")

        for env, env_state in zip(self.envs, env_states):
            env.load_state_dict(env_state)

        loaded_obs = state["current_obs"]
        self.current_obs = [
            GraphObservation.from_dict(obs_dict) if obs_dict is not None else None
            for obs_dict in loaded_obs
        ]

    def get_observations(self) -> List[GraphObservation]:
        obs = []
        for item in self.current_obs:
            if item is None:
                raise RuntimeError("Environment observations are not initialized")
            obs.append(item)
        return obs
