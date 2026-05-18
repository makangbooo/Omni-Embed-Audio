"""Logistic–softplus objective used by ColAF (ColPali-inspired)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class LogisticSoftplusConfig:
    temperature: float = 0.07
    symmetric: bool = False
    reduction: str = "mean"  # 'mean' or 'sum'
    hard_negative_ratio: Optional[float] = None  # If set, use only top-K% hardest negatives
    min_hard_negatives: int = 10  # Minimum number of hard negatives to use


class LogisticSoftplusLoss(torch.nn.Module):
    """
    Implements the logistic-softplus loss applied to a score matrix where the
    diagonal represents positive pairs.

    Given scores S (queries x docs), the loss per query is:
        softplus(-s_pos / τ) + mean_j softplus(s_neg_j / τ)
    """

    def __init__(self, config: Optional[LogisticSoftplusConfig] = None):
        super().__init__()
        self.config = config or LogisticSoftplusConfig()
        if self.config.reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum'")

    def forward(self, scores: torch.Tensor, symmetric: Optional[bool] = None) -> torch.Tensor:
        if scores.dim() != 2 or scores.size(0) != scores.size(1):
            raise ValueError("Expected square 2D score matrix (batch x batch).")
        if scores.size(0) < 2:
            raise ValueError("At least two samples are required to compute negatives.")

        sym = self.config.symmetric if symmetric is None else symmetric

        loss = self._compute(scores)
        if sym:
            loss_t = self._compute(scores.t())
            loss = 0.5 * (loss + loss_t)

        if self.config.reduction == "mean":
            return loss.mean()
        return loss.sum()

    def _compute(self, scores: torch.Tensor) -> torch.Tensor:
        scaled = scores / self.config.temperature
        diag = scaled.diagonal()
        idx = torch.arange(scaled.size(0), device=scores.device)
        mask = torch.ones_like(scaled, dtype=torch.bool)
        mask[idx, idx] = False
        negatives = scaled[mask].view(scaled.size(0), -1)

        # Hard negative mining: select only top-K hardest negatives
        if self.config.hard_negative_ratio is not None and self.config.hard_negative_ratio > 0:
            num_negatives = negatives.size(1)
            k = max(
                self.config.min_hard_negatives,
                int(num_negatives * self.config.hard_negative_ratio)
            )
            k = min(k, num_negatives)  # Don't exceed available negatives

            # Select top-k hardest negatives (highest scores)
            hard_negatives, _ = torch.topk(negatives, k=k, dim=1)
            negatives = hard_negatives

        pos_loss = F.softplus(-diag)
        neg_loss = F.softplus(negatives).mean(dim=-1)
        return pos_loss + neg_loss
