"""Resident BB-RL inference using the evaluated policy and observation code."""
from __future__ import annotations

import hashlib
import time
from typing import Any, Dict
from pathlib import Path

import numpy as np
import torch

from .backbone_init import normalize_init_metadata
from .checkpoint import load_checkpoint
from .config import Config, load_config
from .curriculum import CurriculumScheduler
from .determinism import apply_determinism
from .env import GraphEnv
from .graph_math import edge_count, normalized_density
from .model import BBPolicyNetwork, build_model


ROOT = Path(__file__).resolve().parents[1]


class PolicyRunner:
    """Load once, then greedily complete each supplied connected backbone.

    The checkpoint retains its original ``lite_v3`` format identifier. It is the
    sole BB-RL policy in this release; loading does not modify checkpoint bytes.
    """

    def __init__(self, config_path=None, checkpoint_path=None):
        config_path = Path(config_path) if config_path else ROOT / "configs/bbrl.yaml"
        checkpoint_path = Path(checkpoint_path) if checkpoint_path else ROOT / "checkpoints/bbrl.pt"
        self.cfg = load_config(config_path)
        apply_determinism(self.cfg.determinism)
        self.device = torch.device("cpu")
        checkpoint = load_checkpoint(str(checkpoint_path), map_location="cpu")
        if checkpoint.get("variant", "lite_v3") != "lite_v3":
            raise ValueError("The checkpoint does not contain a BB-RL MLP policy")
        self.model = build_model(self.cfg, self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.env = GraphEnv(
            env_id=0, scheduler=CurriculumScheduler(self.cfg.curriculum),
            top_k=self.cfg.env.top_k, dist_cap=self.cfg.env.dist_cap,
            terminal_bonus_coef=self.cfg.env.terminal_bonus_coef,
            seed=2701, rl_variant="lite_v3", compute_spectral_each_step=False,
            inference_only=self.cfg.policy.fast_inference,
            lite_v3_node_budget=self.cfg.policy.node_budget,
            lite_v3_pair_budget=self.cfg.policy.pair_budget,
            lite_v3_pair_budget_min=self.cfg.policy.pair_budget_min,
        )
        self.provenance = {
            "config": str(config_path), "checkpoint": str(checkpoint_path),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            "method": "BB-RL", "checkpoint_format": "lite_v3",
            "training_steps_at_checkpoint": checkpoint.get("trainer_state", {}).get("global_env_steps"),
            "train_init_mode": self.cfg.init.mode,
        }

    def complete(self, adj: np.ndarray, m: int, branch: dict):
        """Return the completed adjacency and its terminal algebraic connectivity.

        The returned adjacency belongs to this resident environment. Copy it
        before the next call if retaining multiple completed branches.
        """
        n = len(adj)
        info = normalize_init_metadata({
            "init_family": branch["family"], "init_route_label": branch["route"],
            "init_edges": edge_count(adj), "init_lambda2": branch["lambda2"],
            "init_build_runtime_sec": 0.0,
        }, n=n, init_mode="backbone")
        result = _rollout_once(
            model=self.model, selector=None, env=self.env, n=n,
            rho_target=normalized_density(n, m), m_target=m, device=self.device,
            cfg=self.cfg, init_info=info, initial_adj=adj,
        )
        return self.env.adj, float(result["terminal_lambda2"])


def _select_greedy_action_pair(cfg, model, selector, obs, device):
    # Preserve the measured rollout's per-step model timer and greedy tie rule.
    model_start = time.perf_counter()
    output = model.forward_observation(obs, device=device, compute_value=False)
    action_idx = int(torch.argmax(output.logits).item())
    return tuple(int(x) for x in obs.candidate_pairs[action_idx]), float(
        time.perf_counter() - model_start
    )


def _rollout_once(
    *,
    model: BBPolicyNetwork,
    selector: None,
    env: GraphEnv,
    n: int,
    rho_target: float,
    m_target: int,
    device: torch.device,
    cfg: Config,
    init_info: dict[str, Any],
    initial_adj=None,
) -> Dict[str, float | int | str | bool]:
    start = time.perf_counter()
    if initial_adj is None:
        obs = env.reset_with_target(n=n, rho_target=rho_target)
    else:
        obs = env.reset_with_target(
            n=n,
            rho_target=rho_target,
            initial_adj=initial_adj,
            init_mode="backbone",
            init_metadata=init_info,
        )

    done = edge_count(env.adj) >= m_target
    info: Dict[str, Any] = {
        "episode_done": done,
        "episode_len": 0,
        "terminal_lambda2_norm": obs.lambda2_norm,
    }

    time_model_sec = 0.0
    time_env_step_sec = 0.0
    time_obs_build_sec = 0.0
    time_candidate_sec = 0.0
    pool_raw_sum = 0.0
    pool_selected_sum = 0.0
    decision_steps = 0
    while not done:
        pool_raw_sum += float(getattr(obs, "candidate_pool_size_raw", obs.candidate_pairs.shape[0]))
        pool_selected_sum += float(
            getattr(obs, "candidate_pool_size_selected", obs.candidate_pairs.shape[0])
        )
        time_obs_build_sec += float(getattr(obs, "time_obs_build_sec", 0.0))
        time_candidate_sec += float(getattr(obs, "time_candidate_sec", 0.0))
        with torch.inference_mode():
            action_pair, model_sec = _select_greedy_action_pair(
                cfg=cfg,
                model=model,
                selector=selector,
                obs=obs,
                device=device,
            )
        time_model_sec += float(model_sec)
        env_step_start = time.perf_counter()
        obs, _, done, info = env.step(action_pair)
        time_env_step_sec += float(time.perf_counter() - env_step_start)
        decision_steps += 1
    rl_runtime_sec = time.perf_counter() - start

    m_final = edge_count(env.adj)
    lambda2_norm = float(info.get("terminal_lambda2_norm") or 0.0)
    lambda2 = lambda2_norm * float(max(1, n))
    rho_final = normalized_density(n, m_final)
    init_runtime = float(init_info["init_build_runtime_sec"])
    total_runtime = float(rl_runtime_sec + init_runtime)
    denom = max(1, decision_steps)

    return {
        "m_target": int(m_target),
        "m_final": int(m_final),
        "rho_final": float(rho_final),
        "episode_len": int(info.get("episode_len") or 0),
        "terminal_lambda2_norm": float(lambda2_norm),
        "terminal_lambda2": float(lambda2),
        "rl_runtime_sec": float(rl_runtime_sec),
        "runtime_sec": float(rl_runtime_sec),  # Backward-compatible alias.
        "runtime_total_sec": total_runtime,
        "init_mode": str(init_info["init_mode"]),
        "init_route_label": str(init_info["init_route_label"]),
        "init_family": str(init_info["init_family"]),
        "init_edges": int(init_info["init_edges"]),
        "init_lambda2": float(init_info["init_lambda2"]),
        "init_build_runtime_sec": init_runtime,
        "init_cache_hit": bool(init_info["init_cache_hit"]),
        "init_selected_index": str(init_info["init_selected_index"]),
        "init_errors": str(init_info["init_errors"]),
        "candidate_pool_size_raw": float(pool_raw_sum / denom),
        "candidate_pool_size_selected": float(pool_selected_sum / denom),
        "time_obs_build_sec": float(time_obs_build_sec),
        "time_candidate_sec": float(time_candidate_sec),
        "time_model_sec": float(time_model_sec),
        "time_env_step_sec": float(time_env_step_sec),
        "overlap_post_completion_selection": False,
        "overlap_candidate_count": 1,
        "overlap_evaluated_families": str(init_info["init_family"]),
    }
