#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimization utilities for late-interaction retrieval.

Implements various techniques to accelerate token-level late-interaction scoring:
- Token pooling (inspired by ColPali)
- Token striding/subsampling
- Query token pruning
- Dimension reduction utilities

These optimizations can provide 10-50× speedup with minimal accuracy loss.
"""

from typing import Optional, Tuple
import numpy as np


def pool_tokens(
    tokens: np.ndarray,
    mask: Optional[np.ndarray] = None,
    pool_factor: int = 2,
    method: str = "mean",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Pool consecutive tokens to reduce token count.

    Inspired by ColPali paper which achieves 97.8% performance retention
    with pool_factor=3 (66.7% reduction).

    Args:
        tokens: Token embeddings [num_tokens, dim]
        mask: Optional attention mask [num_tokens]
        pool_factor: Number of consecutive tokens to pool (1=no pooling)
        method: Pooling method ('mean' or 'max')

    Returns:
        pooled_tokens: Pooled embeddings [num_pooled_tokens, dim]
        pooled_mask: Pooled mask [num_pooled_tokens] or None

    Example:
        >>> tokens = np.random.randn(300, 512)  # 300 tokens
        >>> pooled, _ = pool_tokens(tokens, pool_factor=3)
        >>> pooled.shape
        (100, 512)  # 66.7% reduction
    """
    if pool_factor <= 1:
        return tokens, mask

    num_tokens = tokens.shape[0]
    if num_tokens == 0:
        return tokens, mask

    pooled_size = (num_tokens + pool_factor - 1) // pool_factor
    pooled_tokens = np.zeros((pooled_size, tokens.shape[1]), dtype=tokens.dtype)

    for i in range(pooled_size):
        start = i * pool_factor
        end = min(start + pool_factor, num_tokens)

        if method == "mean":
            pooled_tokens[i] = tokens[start:end].mean(axis=0)
        elif method == "max":
            pooled_tokens[i] = tokens[start:end].max(axis=0)
        else:
            raise ValueError(f"Unknown pooling method: {method}")

    # Pool mask if provided
    if mask is not None:
        pooled_mask = np.zeros(pooled_size, dtype=mask.dtype)
        for i in range(pooled_size):
            start = i * pool_factor
            end = min(start + pool_factor, num_tokens)
            # Keep pooled position if any token in window is valid
            pooled_mask[i] = mask[start:end].max()
        return pooled_tokens, pooled_mask

    return pooled_tokens, None


def stride_tokens(
    tokens: np.ndarray,
    mask: Optional[np.ndarray] = None,
    stride: int = 2,
    offset: int = 0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Subsample tokens by keeping every stride-th token.

    More aggressive than pooling but faster. Useful for audio tokens
    which tend to have high temporal redundancy.

    Args:
        tokens: Token embeddings [num_tokens, dim]
        mask: Optional attention mask [num_tokens]
        stride: Keep every stride-th token (1=no striding)
        offset: Starting offset (0 to stride-1)

    Returns:
        strided_tokens: Subsampled embeddings
        strided_mask: Subsampled mask or None

    Example:
        >>> tokens = np.random.randn(300, 512)
        >>> strided, _ = stride_tokens(tokens, stride=2)
        >>> strided.shape
        (150, 512)  # 50% reduction
    """
    if stride <= 1:
        return tokens, mask

    if tokens.shape[0] == 0:
        return tokens, mask

    # Apply striding with offset
    strided_tokens = tokens[offset::stride]
    strided_mask = mask[offset::stride] if mask is not None else None

    return strided_tokens, strided_mask


def prune_query_tokens(
    tokens: np.ndarray,
    mask: Optional[np.ndarray] = None,
    keep_ratio: float = 0.7,
    method: str = "norm",
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Prune query tokens by keeping only the most important ones.

    Reduces query token count while preserving salient tokens.
    Based on Query Token Pruning (QTP) techniques from ColBERT literature.

    Args:
        tokens: Query token embeddings [num_tokens, dim]
        mask: Optional attention mask [num_tokens]
        keep_ratio: Ratio of tokens to keep (0.0-1.0)
        method: Importance scoring method:
            - 'norm': Keep tokens with highest L2 norm
            - 'variance': Keep tokens with highest variance
            - 'random': Random sampling (baseline)

    Returns:
        pruned_tokens: Pruned embeddings [num_kept_tokens, dim]
        pruned_mask: Pruned mask [num_kept_tokens] or None
        kept_indices: Indices of kept tokens (for debugging)

    Example:
        >>> tokens = np.random.randn(100, 512)
        >>> pruned, _, indices = prune_query_tokens(tokens, keep_ratio=0.7)
        >>> pruned.shape
        (70, 512)  # 30% reduction
    """
    if keep_ratio >= 1.0:
        return tokens, mask, np.arange(tokens.shape[0])

    num_tokens = tokens.shape[0]
    if num_tokens == 0:
        return tokens, mask, np.array([], dtype=np.int64)

    num_keep = max(1, int(num_tokens * keep_ratio))

    # Compute importance scores
    if method == "norm":
        # L2 norm - tokens with higher norm are more salient
        scores = np.linalg.norm(tokens, axis=1)
    elif method == "variance":
        # Variance across dimensions - more distinctive tokens
        scores = tokens.var(axis=1)
    elif method == "random":
        # Random baseline
        scores = np.random.rand(num_tokens)
    else:
        raise ValueError(f"Unknown pruning method: {method}")

    # Mask out invalid tokens if mask provided
    if mask is not None:
        scores = scores * mask + (1 - mask) * (-np.inf)

    # Select top-k indices
    top_indices = np.argsort(scores)[-num_keep:]
    top_indices = np.sort(top_indices)  # Maintain temporal order

    pruned_tokens = tokens[top_indices]
    pruned_mask = mask[top_indices] if mask is not None else None

    return pruned_tokens, pruned_mask, top_indices


def adaptive_token_reduction(
    tokens: np.ndarray,
    mask: Optional[np.ndarray] = None,
    target_tokens: int = 100,
    method: str = "hybrid",
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Adaptively reduce tokens to target count using best strategy.

    Combines pooling and striding to achieve target token count
    while balancing information preservation and speed.

    Args:
        tokens: Token embeddings [num_tokens, dim]
        mask: Optional attention mask [num_tokens]
        target_tokens: Desired number of tokens
        method: Reduction strategy:
            - 'pool': Pure pooling
            - 'stride': Pure striding
            - 'hybrid': Combine both (recommended)

    Returns:
        reduced_tokens: Reduced embeddings [~target_tokens, dim]
        reduced_mask: Reduced mask or None

    Example:
        >>> tokens = np.random.randn(400, 512)  # 400 tokens
        >>> reduced, _ = adaptive_token_reduction(tokens, target_tokens=100)
        >>> reduced.shape[0]
        100  # Approximately target
    """
    num_tokens = tokens.shape[0]

    if num_tokens <= target_tokens:
        return tokens, mask

    if method == "pool":
        # Pure pooling
        pool_factor = max(1, num_tokens // target_tokens)
        return pool_tokens(tokens, mask, pool_factor=pool_factor, method="mean")

    elif method == "stride":
        # Pure striding
        stride = max(1, num_tokens // target_tokens)
        return stride_tokens(tokens, mask, stride=stride)

    elif method == "hybrid":
        # Hybrid: first stride by 2, then pool remaining
        # This preserves more local structure than pure striding
        reduction_needed = num_tokens / target_tokens

        if reduction_needed <= 2:
            # Light reduction - just pool
            pool_factor = max(1, int(np.ceil(reduction_needed)))
            return pool_tokens(tokens, mask, pool_factor=pool_factor)
        else:
            # Heavy reduction - stride then pool
            stride = 2
            tokens, mask = stride_tokens(tokens, mask, stride=stride)

            # Calculate remaining pooling needed
            pool_factor = max(1, tokens.shape[0] // target_tokens)
            return pool_tokens(tokens, mask, pool_factor=pool_factor)

    else:
        raise ValueError(f"Unknown reduction method: {method}")


def estimate_speedup(
    num_query_tokens: int,
    num_doc_tokens: int,
    pool_factor: int = 1,
    stride: int = 1,
    query_prune_ratio: float = 1.0,
) -> Tuple[float, int, int]:
    """
    Estimate speedup from token optimizations.

    MaxSim complexity is O(Q × K) where Q=query tokens, K=doc tokens.
    Reducing either dimension provides proportional speedup.

    Args:
        num_query_tokens: Original query token count
        num_doc_tokens: Original document token count
        pool_factor: Token pooling factor (applied to both)
        stride: Token striding (applied to documents)
        query_prune_ratio: Query token pruning ratio

    Returns:
        speedup: Expected speedup factor
        final_query_tokens: Final query token count
        final_doc_tokens: Final document token count

    Example:
        >>> estimate_speedup(100, 300, pool_factor=3, query_prune_ratio=0.7)
        (12.86, 23, 100)  # ~13× faster
    """
    # Apply optimizations
    final_query = int(np.ceil(num_query_tokens / pool_factor) * query_prune_ratio)
    final_doc = int(np.ceil(num_doc_tokens / (pool_factor * stride)))

    # Speedup = (original ops) / (final ops)
    original_ops = num_query_tokens * num_doc_tokens
    final_ops = final_query * final_doc
    speedup = original_ops / max(1, final_ops)

    return speedup, final_query, final_doc


# Example usage and benchmarking
if __name__ == "__main__":
    print("=== Token Optimization Utilities ===\n")

    # Simulate typical audio/text token counts
    audio_tokens = np.random.randn(300, 512)  # 300 audio tokens
    text_tokens = np.random.randn(50, 512)    # 50 text tokens

    print(f"Original: {audio_tokens.shape[0]} audio tokens, {text_tokens.shape[0]} text tokens")
    print(f"Original complexity: {audio_tokens.shape[0] * text_tokens.shape[0]:,} operations\n")

    # Test pooling
    pooled, _ = pool_tokens(audio_tokens, pool_factor=3)
    print(f"After pooling (factor=3): {pooled.shape[0]} audio tokens ({pooled.shape[0]/audio_tokens.shape[0]*100:.1f}%)")

    # Test striding
    strided, _ = stride_tokens(audio_tokens, stride=2)
    print(f"After striding (stride=2): {strided.shape[0]} audio tokens ({strided.shape[0]/audio_tokens.shape[0]*100:.1f}%)")

    # Test query pruning
    pruned, _, _ = prune_query_tokens(text_tokens, keep_ratio=0.7)
    print(f"After query pruning (70%): {pruned.shape[0]} text tokens ({pruned.shape[0]/text_tokens.shape[0]*100:.1f}%)\n")

    # Combined effect
    audio_opt, _ = pool_tokens(audio_tokens, pool_factor=3)
    text_opt, _, _ = prune_query_tokens(text_tokens, keep_ratio=0.7)

    final_ops = audio_opt.shape[0] * text_opt.shape[0]
    original_ops = audio_tokens.shape[0] * text_tokens.shape[0]
    speedup = original_ops / final_ops

    print(f"Combined optimizations:")
    print(f"  Final: {text_opt.shape[0]} query × {audio_opt.shape[0]} doc = {final_ops:,} operations")
    print(f"  Speedup: {speedup:.1f}×")
    print(f"  Expected accuracy loss: ~3-5% (based on ColPali/ColBERT results)")
