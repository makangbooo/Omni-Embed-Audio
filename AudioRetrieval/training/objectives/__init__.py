"""Losses and scoring utilities for projection-head training."""

from .maxsim import (
    apply_token_mask,
    compute_maxsim_scores,
    filter_prefix_tokens,
    trim_tokens,
)
from .logistic_softplus import LogisticSoftplusConfig, LogisticSoftplusLoss
from .diffusion import (
    build_diffusion_schedule,
    diffusion_kl_loss,
    sample_noised_scores,
    sample_timesteps,
)

__all__ = [
    "apply_token_mask",
    "compute_maxsim_scores",
    "filter_prefix_tokens",
    "trim_tokens",
    "LogisticSoftplusConfig",
    "LogisticSoftplusLoss",
    "build_diffusion_schedule",
    "diffusion_kl_loss",
    "sample_noised_scores",
    "sample_timesteps",
]
