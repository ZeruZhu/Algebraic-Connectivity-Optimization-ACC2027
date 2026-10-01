import hashlib
from pathlib import Path

import numpy as np

from rl_train.graph_math import build_path_adjacency, laplacian
from rl_train.inference import PolicyRunner


def test_paper_checkpoint_bytes_and_deterministic_inference():
    checkpoint = Path(__file__).resolve().parents[1] / "checkpoints/bbrl.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == (
        "90ace2cca8cfc449486dd36161f82cb3f5797adf32873e0696c51d8a7721dfa8"
    )
    runner = PolicyRunner()
    assert runner.provenance["training_steps_at_checkpoint"] == 30000
    for n in (8, 16, 32, 64, 128):
        initial = build_path_adjacency(n)
        branch = {"family": "path", "route": "test", "lambda2": 0.0}
        first, score = runner.complete(initial, n + 3, branch)
        first = first.copy()
        second, score_again = runner.complete(initial, n + 3, branch)
        np.testing.assert_array_equal(first, second)
        assert score == score_again
        np.testing.assert_allclose(score, np.linalg.eigvalsh(laplacian(first))[1], atol=1e-10)
