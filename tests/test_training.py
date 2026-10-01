import numpy as np
import torch

from rl_train.checkpoint import load_checkpoint
from rl_train.config import CurriculumConfig, CurriculumPhaseConfig, load_config
from rl_train.graph_math import build_path_adjacency, edge_count
from rl_train.inference import PolicyRunner
from rl_train.train_bbrl import run_training


def test_training_updates_and_produces_loadable_policy(tmp_path):
    cfg = load_config()
    cfg.env.num_envs = 2
    cfg.curriculum = CurriculumConfig([CurriculumPhaseConfig("smoke", 0, 800000, 8, 8)])
    cfg.backbone_init.cayley_samples = 8
    cfg.ppo.rollout_env_steps = 8
    cfg.ppo.minibatch_size = 4
    cfg.ppo.update_epochs = 1
    cfg.evaluation.episodes = 1
    cfg.train.eval_every_env_steps = 8
    cfg.train.checkpoint_every_env_steps = 8
    out = tmp_path / "training"
    result = run_training(cfg, out, stop_after=16)
    assert result["update_idx"] == 2
    assert (out / "best.pt").exists()
    checkpoint = load_checkpoint(str(out / "latest.pt"), map_location="cpu")
    assert checkpoint["trainer_state"]["global_env_steps"] == 16
    assert all(torch.isfinite(t).all() for t in checkpoint["model_state"].values())
    runner = PolicyRunner(checkpoint_path=out / "latest.pt")
    initial = build_path_adjacency(8)
    graph, score = runner.complete(initial, 10, {"family": "path", "route": "test", "lambda2": 0.0})
    assert edge_count(graph) == 10
    assert np.all(graph >= initial)
    assert score > 0
