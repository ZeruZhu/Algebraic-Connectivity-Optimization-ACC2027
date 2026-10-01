from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .env import GraphObservation
from .model import BBPolicyNetwork


@dataclass
class TransitionRecord:
    obs: Dict
    action_idx: int
    old_logprob: float
    old_value: float
    reward: float
    done: bool

    def to_dict(self) -> Dict:
        return {
            "obs": self.obs,
            "action_idx": self.action_idx,
            "old_logprob": self.old_logprob,
            "old_value": self.old_value,
            "reward": self.reward,
            "done": self.done,
        }

    @staticmethod
    def from_dict(data: Dict) -> "TransitionRecord":
        return TransitionRecord(
            obs=data["obs"],
            action_idx=int(data["action_idx"]),
            old_logprob=float(data["old_logprob"]),
            old_value=float(data["old_value"]),
            reward=float(data["reward"]),
            done=bool(data["done"]),
        )


@dataclass
class PPOSample:
    obs: Dict
    action_idx: int
    old_logprob: float
    advantage: float
    return_t: float


class RolloutBuffer:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.transitions_by_env: List[List[TransitionRecord]] = [[] for _ in range(num_envs)]
        self.total_transitions: int = 0

    def add(self, env_idx: int, record: TransitionRecord) -> None:
        self.transitions_by_env[env_idx].append(record)
        self.total_transitions += 1

    def __len__(self) -> int:
        return self.total_transitions

    def clear(self) -> None:
        self.transitions_by_env = [[] for _ in range(self.num_envs)]
        self.total_transitions = 0

    def state_dict(self) -> Dict:
        return {
            "num_envs": self.num_envs,
            "total_transitions": self.total_transitions,
            "transitions_by_env": [
                [record.to_dict() for record in env_records]
                for env_records in self.transitions_by_env
            ],
        }

    def load_state_dict(self, state: Dict) -> None:
        if int(state["num_envs"]) != self.num_envs:
            raise ValueError("Mismatched num_envs in rollout buffer state")
        self.total_transitions = int(state["total_transitions"])
        self.transitions_by_env = [
            [TransitionRecord.from_dict(record_dict) for record_dict in env_records]
            for env_records in state["transitions_by_env"]
        ]

    def build_samples(
        self,
        next_values: Sequence[float],
        gamma: float,
        gae_lambda: float,
    ) -> List[PPOSample]:
        samples: List[PPOSample] = []

        for env_idx, env_records in enumerate(self.transitions_by_env):
            if not env_records:
                continue

            env_samples_rev: List[PPOSample] = []
            next_val = float(next_values[env_idx])
            gae = 0.0

            for record in reversed(env_records):
                not_done = 0.0 if record.done else 1.0
                delta = record.reward + gamma * next_val * not_done - record.old_value
                gae = delta + gamma * gae_lambda * not_done * gae
                return_t = gae + record.old_value

                env_samples_rev.append(
                    PPOSample(
                        obs=record.obs,
                        action_idx=record.action_idx,
                        old_logprob=record.old_logprob,
                        advantage=gae,
                        return_t=return_t,
                    )
                )

                next_val = record.old_value

            samples.extend(reversed(env_samples_rev))

        return samples


def ppo_update(
    model: BBPolicyNetwork,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    buffer: RolloutBuffer,
    next_values: Sequence[float],
    gamma: float,
    gae_lambda: float,
    clip_ratio: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    update_epochs: int,
    minibatch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    samples = buffer.build_samples(next_values=next_values, gamma=gamma, gae_lambda=gae_lambda)
    if not samples:
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }

    advantages = np.array([s.advantage for s in samples], dtype=np.float32)
    adv_mean = float(np.mean(advantages))
    adv_std = float(np.std(advantages) + 1e-8)
    advantages = (advantages - adv_mean) / adv_std

    num_samples = len(samples)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_kl = 0.0
    total_batches = 0

    for _ in range(update_epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)

        for start in range(0, num_samples, minibatch_size):
            batch_idx = indices[start : start + minibatch_size]
            if len(batch_idx) == 0:
                continue

            policy_losses = []
            value_losses = []
            entropies = []
            kls = []

            for idx in batch_idx:
                sample = samples[int(idx)]
                obs = GraphObservation.from_dict(sample.obs)
                output = model.forward_observation(obs, device=device)
                dist = Categorical(logits=output.logits)

                action_t = torch.tensor(sample.action_idx, dtype=torch.long, device=device)
                old_logprob_t = torch.tensor(sample.old_logprob, dtype=torch.float32, device=device)
                advantage_t = torch.tensor(float(advantages[int(idx)]), dtype=torch.float32, device=device)
                return_t = torch.tensor(sample.return_t, dtype=torch.float32, device=device)

                new_logprob = dist.log_prob(action_t)
                entropy = dist.entropy()

                ratio = torch.exp(new_logprob - old_logprob_t)
                surr1 = ratio * advantage_t
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage_t
                policy_loss = -torch.min(surr1, surr2)

                value_loss = F.mse_loss(output.value, return_t)
                kl = old_logprob_t - new_logprob

                policy_losses.append(policy_loss)
                value_losses.append(value_loss)
                entropies.append(entropy)
                kls.append(kl)

            mean_policy_loss = torch.stack(policy_losses).mean()
            mean_value_loss = torch.stack(value_losses).mean()
            mean_entropy = torch.stack(entropies).mean()
            mean_kl = torch.stack(kls).mean()

            loss = mean_policy_loss + value_coef * mean_value_loss - entropy_coef * mean_entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_policy_loss += float(mean_policy_loss.detach().cpu().item())
            total_value_loss += float(mean_value_loss.detach().cpu().item())
            total_entropy += float(mean_entropy.detach().cpu().item())
            total_kl += float(mean_kl.detach().cpu().item())
            total_batches += 1

    scheduler.step()

    denom = max(1, total_batches)
    return {
        "policy_loss": total_policy_loss / denom,
        "value_loss": total_value_loss / denom,
        "entropy": total_entropy / denom,
        "approx_kl": total_kl / denom,
        "learning_rate": optimizer.param_groups[0]["lr"],
    }
