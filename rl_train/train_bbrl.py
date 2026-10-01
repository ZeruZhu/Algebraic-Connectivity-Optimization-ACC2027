"""Train BB-RL with PPO using the published policy, rewards, and curriculum.

This entry point is a cleaned training driver. The released paper results use
the supplied checkpoint, not a retrained model produced by this driver.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical

from .backbone_init import BackboneInitializer
from .checkpoint import save_checkpoint
from .config import load_config
from .curriculum import CurriculumScheduler
from .determinism import apply_determinism, get_rng_state
from .env import GraphEnv, VectorGraphEnvManager
from .graph_math import edge_count
from .model import build_model
from .ppo import RolloutBuffer, TransitionRecord, ppo_update


def evaluate_policy(cfg, model, curriculum, device, steps, initializer):
    model.eval()
    env = GraphEnv(
        env_id=999, scheduler=curriculum, top_k=cfg.env.top_k,
        dist_cap=cfg.env.dist_cap, terminal_bonus_coef=cfg.env.terminal_bonus_coef,
        seed=cfg.determinism.seed + 99991, compute_spectral_each_step=True,
        lite_v3_node_budget=cfg.policy.node_budget,
        lite_v3_pair_budget=cfg.policy.pair_budget,
        lite_v3_pair_budget_min=cfg.policy.pair_budget_min,
    )
    scores = []
    for _ in range(max(1, cfg.evaluation.episodes)):
        obs = env.reset(steps, init_mode=cfg.init.mode,
                        initial_adj_builder=initializer.get_initial_adj if initializer else None)
        done = edge_count(env.adj) >= env.m_target
        while not done:
            with torch.no_grad():
                output = model.forward_observation(obs, device)
                pair = tuple(int(x) for x in obs.candidate_pairs[int(output.logits.argmax())])
            obs, _, done, _ = env.step(pair)
        scores.append(obs.lambda2_norm)
    model.train()
    return float(np.mean(scores))


def run_training(cfg, out_dir, *, stop_after=None, device="cpu"):
    """Run fresh training; stop_after leaves the original LR horizon unchanged."""
    apply_determinism(cfg.determinism)
    steps_limit = cfg.train.max_env_steps if stop_after is None else int(stop_after)
    if not 0 < steps_limit <= cfg.train.max_env_steps or steps_limit % cfg.env.num_envs:
        raise ValueError("Training steps must be positive, within the schedule, and divisible by num_envs")
    if cfg.train.eval_every_env_steps < 1 or cfg.train.checkpoint_every_env_steps < 1:
        raise ValueError("Evaluation and checkpoint intervals must be positive")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(device)
    curriculum = CurriculumScheduler(cfg.curriculum)
    initializer = BackboneInitializer(cfg.backbone_init) if cfg.init.mode == "backbone" else None
    envs = VectorGraphEnvManager(
        num_envs=cfg.env.num_envs, scheduler=curriculum, top_k=cfg.env.top_k,
        dist_cap=cfg.env.dist_cap, terminal_bonus_coef=cfg.env.terminal_bonus_coef,
        base_seed=cfg.determinism.seed, compute_spectral_each_step=True,
        lite_v3_node_budget=cfg.policy.node_budget,
        lite_v3_pair_budget=cfg.policy.pair_budget,
        lite_v3_pair_budget_min=cfg.policy.pair_budget_min,
        init_mode=cfg.init.mode,
        initial_adj_builder=initializer.get_initial_adj if initializer else None,
    )
    model = build_model(cfg, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.ppo.learning_rate)
    total_updates = max(1, cfg.train.max_env_steps // cfg.ppo.rollout_env_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda i: max(0.0, 1.0 - i / total_updates))
    buffer = RolloutBuffer(cfg.env.num_envs)
    envs.reset_all(0)
    steps, update_idx, last_eval, last_checkpoint = 0, 0, 0, 0
    best_score = float("-inf")

    def payload():
        return {
            "variant": "lite_v3", "method": "BB-RL", "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "config": asdict(cfg), "rng_state": get_rng_state(),
            "env_manager_state": envs.state_dict(), "rollout_buffer_state": buffer.state_dict(),
            "trainer_state": {"global_env_steps": steps, "update_idx": update_idx,
                              "best_eval_metric": best_score},
            "driver": "ACC2027 cleaned BB-RL PPO training driver",
        }

    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["global_env_steps", "update_idx", "policy_loss", "value_loss", "entropy",
                  "approx_kl", "learning_rate"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        while steps < steps_limit:
            observations = envs.get_observations()
            actions, pairs, logprobs, values = [], [], [], []
            with torch.no_grad():
                for obs in observations:
                    output = model.forward_observation(obs, device)
                    dist = Categorical(logits=output.logits)
                    action = dist.sample()
                    actions.append(int(action.item()))
                    pairs.append(tuple(int(x) for x in obs.candidate_pairs[actions[-1]]))
                    logprobs.append(float(dist.log_prob(action).item()))
                    values.append(float(output.value.item()))
            steps += cfg.env.num_envs
            next_obs, rewards, dones, _ = envs.step(pairs, steps)
            for i in range(cfg.env.num_envs):
                buffer.add(i, TransitionRecord(observations[i].to_dict(), actions[i], logprobs[i],
                                               values[i], float(rewards[i]), bool(dones[i])))
            if len(buffer) >= cfg.ppo.rollout_env_steps:
                with torch.no_grad():
                    next_values = [float(model.forward_observation(obs, device).value.item()) for obs in next_obs]
                params = asdict(cfg.ppo)
                params.pop("learning_rate")
                params.pop("rollout_env_steps")
                metrics = ppo_update(model, optimizer, scheduler, buffer, next_values,
                                     device=device, **params)
                buffer.clear()
                update_idx += 1
                writer.writerow(dict(global_env_steps=steps, update_idx=update_idx, **metrics))
                stream.flush()
                print(f"steps={steps} updates={update_idx} policy_loss={metrics['policy_loss']:.5f}", flush=True)
            if steps - last_eval >= cfg.train.eval_every_env_steps:
                score = evaluate_policy(cfg, model, curriculum, device, steps, initializer)
                last_eval = steps
                if score > best_score:
                    best_score = score
                    save_checkpoint(str(out_dir / "best.pt"), payload())
            if steps - last_checkpoint >= cfg.train.checkpoint_every_env_steps:
                save_checkpoint(str(out_dir / "latest.pt"), payload())
                last_checkpoint = steps
        save_checkpoint(str(out_dir / "latest.pt"), payload())
    return {"global_env_steps": steps, "update_idx": update_idx, "output_dir": str(out_dir)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, help="Stop early without changing the configured learning-rate horizon")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(run_training(load_config(args.config), args.out_dir, stop_after=args.steps, device=args.device))


if __name__ == "__main__":
    main()
