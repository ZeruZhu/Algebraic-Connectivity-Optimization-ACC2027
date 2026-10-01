from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DesignConfig:
    rho_very_low: float = 0.05
    rho_low_mix_end: float = 0.10
    rho_low: float = 0.5
    rho_high: float = 0.75
    routing_mode: str = "window"  # window, always_both, always_envelope, always_cayley
    cayley_samples: int = 500
    cayley_degree_slack: int = 0
    cayley_multi_index_overlap: bool = False
    cayley_enable_index4: bool = True
    cayley_sampling_mode: str = "coset"  # coset, random
    cayley_index2_group: str = "current"  # current (semidirect), cyclic
    cayley_independent_index_seeds: bool = False
    cayley_index2_overlap_high: float = 0.6
    cayley_index3_overlap_high: float = 2.0 / 3.0
    cayley_index4_overlap_low: float = 2.0 / 3.0
    cayley_eval_batch_size: int = 128
    cayley_spectral_backend: str = "cpu"  # cpu, cuda (optional)
    cayley_eval_mode: str = "dense"  # dense, character
    cayley_dense_solver: str = "full"  # full, subset, fft_iterative (cyclic deletion only)
    force_connected_backbone: bool = True
