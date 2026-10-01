from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import yaml


@dataclass
class CurriculumPhaseConfig:
    name: str
    start_env_step: int
    end_env_step: int
    n_min: int
    n_max: int


@dataclass
class CurriculumConfig:
    phases: List[CurriculumPhaseConfig] = field(default_factory=list)


@dataclass
class EnvConfig:
    num_envs: int = 8
    top_k: int = 128
    dist_cap: int = 4
    terminal_bonus_coef: float = 0.2


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    learning_rate: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 1.0
    update_epochs: int = 4
    minibatch_size: int = 64
    rollout_env_steps: int = 2048


@dataclass
class DeterminismConfig:
    strict: bool = True
    seed: int = 1234
    num_threads: int = 1


@dataclass
class EvalConfig:
    episodes: int = 8


@dataclass
class InitConfig:
    mode: str = "path"  # path, backbone


@dataclass
class BackboneInitConfig:
    routing_mode: str = "window"  # window, always_both, always_envelope, always_cayley
    rho_very_low: float = 0.05
    rho_low_mix_end: float = 0.10
    rho_low: float = 0.5
    rho_high: float = 0.75
    cayley_samples: int = 500
    cayley_degree_slack: int = 0
    cayley_multi_index_overlap: bool = False
    cayley_enable_index4: bool = True
    cayley_index2_overlap_high: float = 0.6
    cayley_index3_overlap_high: float = 2.0 / 3.0
    cayley_index4_overlap_low: float = 2.0 / 3.0
    cayley_spectral_backend: str = "cpu"  # cpu, cuda
    cayley_eval_batch_size: int = 128
    cayley_eval_mode: str = "dense"  # dense, character
    force_connected_backbone: bool = True



@dataclass
class PolicyConfig:
    node_budget: int = 32
    pair_budget: int = 96
    pair_budget_min: int = 32
    scorer_hidden_dim: int = 64
    scorer_layers: int = 2
    value_hidden_dim: int = 64
    fast_inference: bool = True

@dataclass
class TrainConfig:
    max_env_steps: int = 800000
    output_dir: str = "runs/bbrl"
    eval_every_env_steps: int = 10000
    checkpoint_every_env_steps: int = 20000

@dataclass
class Config:
    curriculum: CurriculumConfig
    env: EnvConfig
    ppo: PPOConfig
    determinism: DeterminismConfig
    evaluation: EvalConfig
    backbone_init: BackboneInitConfig
    policy: PolicyConfig
    train: TrainConfig
    init: InitConfig


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "configs/bbrl.yaml"
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    constructors = {"env": EnvConfig, "ppo": PPOConfig, "determinism": DeterminismConfig,
                    "evaluation": EvalConfig, "backbone_init": BackboneInitConfig,
                    "policy": PolicyConfig, "train": TrainConfig, "init": InitConfig}
    unknown = set(raw) - set(constructors) - {"curriculum"}
    if unknown:
        raise ValueError(f"Unknown BB-RL config sections: {sorted(unknown)}")
    phases = [CurriculumPhaseConfig(**p) for p in raw["curriculum"]["phases"]]
    if not phases or any(p.n_min < 3 or p.n_max < p.n_min for p in phases):
        raise ValueError("The curriculum requires valid graph-size intervals with n >= 3")
    cfg = Config(curriculum=CurriculumConfig(phases),
                 **{key: cls(**raw.get(key, {})) for key, cls in constructors.items()})
    if cfg.env.num_envs < 1 or cfg.ppo.rollout_env_steps < 1:
        raise ValueError("Environment count and rollout length must be positive")
    if cfg.policy.node_budget < 1 or cfg.policy.pair_budget < 1 or cfg.policy.pair_budget_min < 1:
        raise ValueError("Candidate budgets must be positive")
    if cfg.init.mode not in {"path", "backbone"}:
        raise ValueError("init.mode must be path or backbone")
    return cfg
