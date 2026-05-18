"""Diffusion-based reranker components (DiffATR-inspired)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import nn
from torch.nn import functional as F


def _build_sinusoidal_embedding(dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    if half == 0:
        return torch.zeros(1, 0)
    exponents = torch.arange(half, dtype=torch.float32) / half
    freqs = torch.exp(-math.log(max_period) * exponents)
    return freqs


class SinusoidalTimeEmbedding(nn.Module):
    """
    Deterministic timestep embedding mirroring transformer positional encodings.
    """

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        freqs = _build_sinusoidal_embedding(dim, max_period)  # (dim//2,)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (batch,) integer or float tensor
        Returns:
            (batch, dim) sinusoidal embeddings
        """
        if timesteps.dim() == 0:
            timesteps = timesteps[None]
        freqs = self.freqs  # [dim//2]
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=1)
        if emb.size(-1) < self.dim:
            pad = self.dim - emb.size(-1)
            emb = F.pad(emb, (0, pad))
        return emb


@dataclass
class DiffusionSchedule:
    num_steps: int
    beta_start: float = 1e-3
    beta_end: float = 5e-2
    schedule: Literal["linear", "cosine"] = "linear"

    def build(self, device: Optional[torch.device] = None) -> torch.Tensor:
        if self.schedule == "linear":
            betas = torch.linspace(self.beta_start, self.beta_end, self.num_steps, device=device)
        elif self.schedule == "cosine":
            timesteps = torch.linspace(0, self.num_steps, self.num_steps + 1, device=device)
            alphas_cumprod = torch.cos((timesteps / self.num_steps + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]).clamp(min=1e-5)
        else:
            raise ValueError(f"Unsupported schedule: {self.schedule}")
        return betas.clamp(1e-6, 1 - 1e-6)


class AttentionDenoiser(nn.Module):
    """
    Attention-based denoiser described in DiffATR.

    Given a query embedding and a candidate set, predicts refined probability
    scores conditioned on the noisy distribution x_k.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 512, mlp_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.scale = hidden_dim ** -0.5

        self.q_proj = nn.Linear(feature_dim, hidden_dim)
        self.k_proj = nn.Linear(feature_dim, hidden_dim)
        self.v_proj = nn.Linear(feature_dim, hidden_dim)

        self.time_embed = SinusoidalTimeEmbedding(feature_dim)
        self.time_query = nn.Linear(feature_dim, hidden_dim)
        self.time_key = nn.Linear(feature_dim, hidden_dim)
        self.time_value = nn.Linear(feature_dim, hidden_dim)

        self.context_proj = nn.Linear(hidden_dim, feature_dim)

        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, 1),
        )

    def forward(
        self,
        text_embed: torch.Tensor,
        audio_embed: torch.Tensor,
        x_k: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_embed: (num_queries, dim) - Can be 1 (inference) or N (training)
            audio_embed: (num_candidates, dim) - Can be N or any number
            x_k: (num_queries, num_candidates) noisy scores for each query
            timesteps: (num_queries,) diffusion step indices

        Returns:
            logits predicting the clean distribution (num_queries, num_candidates)
        """
        num_queries, dim = text_embed.shape
        num_candidates, audio_dim = audio_embed.shape

        if audio_dim != dim:
            raise ValueError(f"text_embed and audio_embed must have same dim; got {dim} vs {audio_dim}")
        if x_k.shape != (num_queries, num_candidates):
            raise ValueError(
                f"x_k must be (num_queries, num_candidates); "
                f"got {x_k.shape}, expected ({num_queries}, {num_candidates})"
            )

        t_embed = self.time_embed(timesteps.to(text_embed.device))
        q = self.q_proj(text_embed) + self.time_query(t_embed)  # (num_queries, hidden)

        key_base = self.k_proj(audio_embed)  # (num_candidates, hidden)
        value_base = self.v_proj(audio_embed)  # (num_candidates, hidden)

        time_key = self.time_key(t_embed).unsqueeze(1)  # (num_queries, 1, hidden)
        time_value = self.time_value(t_embed).unsqueeze(1)  # (num_queries, 1, hidden)

        # Expand candidates per query
        k = key_base.unsqueeze(0) + time_key  # (num_queries, num_candidates, hidden)
        v = value_base.unsqueeze(0) + time_value  # (num_queries, num_candidates, hidden)

        attn_logits = torch.matmul(q.unsqueeze(1), k.transpose(-2, -1)).squeeze(1) * self.scale  # (num_queries, num_candidates)
        attn_logits = attn_logits + x_k
        attn = torch.softmax(attn_logits, dim=-1)

        context = (attn.unsqueeze(-1) * v).sum(dim=1)  # (num_queries, hidden)
        context = self.context_proj(context)  # (num_queries, dim)
        context = self.dropout(context)

        audio_expanded = audio_embed.unsqueeze(0).expand(num_queries, num_candidates, dim)
        context_expanded = context.unsqueeze(1).expand(num_queries, num_candidates, dim)
        mlp_input = torch.cat([audio_expanded, context_expanded], dim=-1)  # (num_queries, num_candidates, 2*dim)
        mlp_input = self.dropout(mlp_input)

        logits = self.mlp(mlp_input).squeeze(-1)  # (num_queries, num_candidates)
        return logits


class DiffusionRerankerModel(nn.Module):
    """
    Wrapper combining the attention denoiser with optional projection heads.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.denoiser = AttentionDenoiser(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

    def forward(
        self,
        text_embed: torch.Tensor,
        audio_embed: torch.Tensor,
        x_k: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        return self.denoiser(text_embed, audio_embed, x_k, timesteps)
