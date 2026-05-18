"""Objective helpers for diffusion-based reranker training."""

from __future__ import annotations

from typing import Literal, Tuple

import torch
from torch.nn import functional as F

from AudioRetrieval.training.models.diffusion import DiffusionSchedule


def build_diffusion_schedule(
    num_steps: int,
    device: torch.device,
    beta_start: float = 1e-3,
    beta_end: float = 5e-2,
    schedule: Literal["linear", "cosine"] = "linear",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (betas, alphas, alphas_cumprod) tensors shaped (num_steps,)
    """
    betas = DiffusionSchedule(
        num_steps=num_steps,
        beta_start=beta_start,
        beta_end=beta_end,
        schedule=schedule,
    ).build(device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


def sample_timesteps(batch_size: int, num_steps: int, device: torch.device) -> torch.Tensor:
    """
    Uniformly sample integer diffusion steps in [0, num_steps-1]
    """
    return torch.randint(low=0, high=num_steps, size=(batch_size,), device=device)


def sample_noised_scores(
    x0: torch.Tensor,
    alphas_cumprod: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    """
    Sample noisy distributions x_t given the clean one-hot scores x0.

    Args:
        x0: (batch, num_candidates)
        alphas_cumprod: (num_steps,) cumulative product of alphas
        timesteps: (batch,) integer steps
    """
    alpha_t = alphas_cumprod[timesteps].unsqueeze(-1)
    noise = torch.randn_like(x0)
    return torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise


def diffusion_kl_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    KL-divergence between predicted logits and one-hot targets (equivalent to CE).

    Args:
        logits: (batch, num_candidates)
        targets: (batch,) integer indices representing positives
    """
    return F.cross_entropy(logits, targets)
