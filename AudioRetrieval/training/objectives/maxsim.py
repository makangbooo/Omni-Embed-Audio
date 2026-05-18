"""Utilities for ColAF MaxSim scoring (ColPali-inspired)."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

NEG_INF = -1e4


def apply_token_mask(tokens: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Zero-out invalid tokens using a broadcastable mask (1 = keep, 0 = drop).
    """
    if mask is None:
        return tokens
    return tokens * mask.unsqueeze(-1).to(tokens.dtype)


def filter_prefix_tokens(
    tokens: torch.Tensor,
    mask: Optional[torch.Tensor],
    prefix_length: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Drop the first ``prefix_length`` tokens (used for instruction/prompt tokens).
    """
    if prefix_length <= 0:
        return tokens, mask
    tokens = tokens[..., prefix_length:, :]
    if mask is not None:
        mask = mask[..., prefix_length:]
    return tokens, mask


def trim_tokens(
    tokens: torch.Tensor,
    mask: Optional[torch.Tensor],
    max_tokens: Optional[int],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Retain at most ``max_tokens`` entries along the penultimate dimension.
    """
    if max_tokens is None or tokens.size(-2) <= max_tokens:
        return tokens, mask
    tokens = tokens[..., :max_tokens, :]
    if mask is not None:
        mask = mask[..., :max_tokens]
    return tokens, mask


def compute_maxsim_scores(
    query_tokens: torch.Tensor,
    doc_tokens: torch.Tensor,
    query_mask: Optional[torch.Tensor] = None,
    doc_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute pairwise MaxSim scores between batches of query/doc token sets.

    Args:
        query_tokens: [B, Q, D] tensor (assumed L2-normalised).
        doc_tokens:   [B, K, D] tensor (assumed L2-normalised).
        query_mask:   Optional [B, Q] float/bool mask (1=valid).
        doc_mask:     Optional [B, K] float/bool mask (1=valid).

    Returns:
        [B, B] matrix where entry (i, j) is the MaxSim score of query i against doc j.
    """
    if query_tokens.dim() != 3 or doc_tokens.dim() != 3:
        raise ValueError("Expected 3D tensors for query/doc tokens.")
    if query_tokens.size(0) != doc_tokens.size(0):
        raise ValueError("Batch sizes must match for queries/docs.")
    if query_tokens.size(-1) != doc_tokens.size(-1):
        raise ValueError("Token embedding dimensions must match.")

    B, Q, D = query_tokens.shape
    _, K, _ = doc_tokens.shape

    # Compute pairwise token similarities: [B, B, Q, K]
    scores = torch.matmul(
        query_tokens.unsqueeze(1),               # [B, 1, Q, D]
        doc_tokens.unsqueeze(0).transpose(-1, -2)  # [1, B, D, K]
    )

    if doc_mask is not None:
        doc_mask = doc_mask.to(query_tokens.dtype)
        doc_mask = doc_mask.unsqueeze(0).unsqueeze(2)  # [1, B, 1, K]
        scores = scores.masked_fill(doc_mask == 0, NEG_INF)

    # Max over document tokens -> [B, B, Q]
    max_over_doc = scores.max(dim=-1).values

    if query_mask is not None:
        query_mask = query_mask.to(query_tokens.dtype)
        query_mask = query_mask.unsqueeze(1)  # [B, 1, Q]
        weighted_sum = (max_over_doc * query_mask).sum(dim=-1)
        counts = query_mask.sum(dim=-1).clamp_min(1.0)
        result = weighted_sum / counts
    else:
        result = max_over_doc.mean(dim=-1)

    # Replace any NaNs/Infs arising from fully-masked entries.
    if not torch.isfinite(result).all():
        result = torch.where(torch.isfinite(result), result, torch.zeros_like(result))
    return result
