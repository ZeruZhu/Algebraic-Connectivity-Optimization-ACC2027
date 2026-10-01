from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
from .env import GraphObservation

@dataclass
class ForwardOutput:
    logits: torch.Tensor
    value: torch.Tensor


class BBPolicyNetwork(nn.Module):
    """Single-stage BB-RL policy with a separate training value network."""

    def __init__(
        self,
        node_feature_dim: int,
        pair_feature_dim: int,
        global_feature_dim: int,
        scorer_hidden_dim: int,
        scorer_layers: int,
        value_hidden_dim: int,
    ):
        super().__init__()
        if scorer_layers < 1:
            raise ValueError("scorer_layers must be >= 1")

        rep_dim = (2 * node_feature_dim) + pair_feature_dim + global_feature_dim

        scorer_layers_list: list[nn.Module] = []
        in_dim = rep_dim
        for _ in range(scorer_layers):
            scorer_layers_list.append(nn.Linear(in_dim, scorer_hidden_dim))
            scorer_layers_list.append(nn.ReLU())
            in_dim = scorer_hidden_dim
        scorer_layers_list.append(nn.Linear(in_dim, 1))
        self.pair_scorer = nn.Sequential(*scorer_layers_list)

        self.value_mlp = nn.Sequential(
            nn.Linear(node_feature_dim + global_feature_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, 1),
        )

    def forward_observation(
        self, obs: GraphObservation, device: torch.device, *, compute_value: bool = True,
    ) -> ForwardOutput:
        x = torch.as_tensor(obs.node_features, dtype=torch.float32, device=device)
        candidate_pairs = torch.as_tensor(obs.candidate_pairs, dtype=torch.long, device=device)
        pair_features = torch.as_tensor(obs.pair_features, dtype=torch.float32, device=device)
        global_features = torch.as_tensor(obs.global_features, dtype=torch.float32, device=device)

        if candidate_pairs.numel() == 0:
            raise RuntimeError("No candidate actions available for the current graph state")

        i_idx = candidate_pairs[:, 0]
        j_idx = candidate_pairs[:, 1]
        x_i = x[i_idx]
        x_j = x[j_idx]
        rep = torch.cat(
            [
                torch.abs(x_i - x_j),
                x_i * x_j,
                pair_features,
                global_features.unsqueeze(0).expand(candidate_pairs.shape[0], -1),
            ],
            dim=1,
        )

        logits = self.pair_scorer(rep).squeeze(-1)
        if not compute_value:
            return ForwardOutput(logits=logits, value=logits.new_zeros(()))
        pooled_x = torch.mean(x, dim=0)
        value_in = torch.cat([pooled_x, global_features], dim=0)
        value = self.value_mlp(value_in).squeeze(-1)
        return ForwardOutput(logits=logits, value=value)


def build_model(cfg, device):
    return BBPolicyNetwork(3, 5, 3, cfg.policy.scorer_hidden_dim, cfg.policy.scorer_layers, cfg.policy.value_hidden_dim).to(device)
