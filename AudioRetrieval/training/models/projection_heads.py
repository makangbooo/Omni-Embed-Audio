"""
Projection heads for ColAF late-interaction training (inspired by ColPali).

Each head is a lightweight linear projection followed by LayerNorm (and
optional dropout). Tokens are typically L2-normalised after passing through the
head so cosine similarity reduces to a dot product.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Row-wise L2 normalisation supporting arbitrary leading dimensions.
    """
    norm = x.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    return x / norm


@dataclass
class ProjectionConfig:
    input_dim: int
    output_dim: int
    bias: bool = False
    dropout: float = 0.0


class TokenProjectionHead(nn.Module):
    """
    Single linear projection + normalisation step used for both audio and text
    tokens. The caller is responsible for applying L2 normalisation if desired.
    """

    def __init__(self, config: ProjectionConfig):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.input_dim, config.output_dim, bias=config.bias)
        self.norm = nn.LayerNorm(config.output_dim)
        self.dropout: nn.Module
        if config.dropout > 0.0:
            self.dropout = nn.Dropout(config.dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (..., input_dim)

        Returns:
            (..., output_dim) tensor after projection + normalisation.
        """
        out = self.proj(tokens)
        out = self.norm(out)
        out = self.dropout(out)
        return out


class ColAF_MLP_Deep(nn.Module):
    """
    Improved projection head with deeper architecture, residual connections,
    and larger projection dimension for better information preservation.

    Architecture:
        Input (3584) → LayerNorm → Linear(3584→1024) → LayerNorm+Residual
        → Dropout→GELU → Linear(1024→512) → LayerNorm+Residual
        → Dropout→GELU → Linear(512→512)

    Benefits over baseline:
    - Larger projection (512 vs 256) = less information loss
    - Deeper (3 layers vs 2) = more expressiveness
    - Residual connections = better gradient flow
    - Progressive dimension reduction: 3584 → 1024 → 512 → 512
    """

    def __init__(
        self,
        input_dim: int = 3584,
        hidden_dim: int = 1024,
        output_dim: int = 512,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Input layer norm
        self.input_norm = nn.LayerNorm(input_dim)

        # Layer 1: 3584 → 1024
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.activation1 = nn.GELU()

        # Layer 2: 1024 → 512
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.norm2 = nn.LayerNorm(output_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.activation2 = nn.GELU()

        # Layer 3: 512 → 512 (refinement)
        self.fc3 = nn.Linear(output_dim, output_dim, bias=bias)

        # Residual projection (for skip connection from layer 1 to layer 2)
        self.residual_proj = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (..., input_dim)

        Returns:
            (..., output_dim) tensor after deep projection
        """
        # Input normalization
        x = self.input_norm(tokens)

        # Layer 1
        x = self.fc1(x)
        x = self.norm1(x)
        hidden = x  # Save for residual
        x = self.dropout1(x)
        x = self.activation1(x)

        # Layer 2 with residual
        x = self.fc2(x)
        x = x + self.residual_proj(hidden)  # Residual connection
        x = self.norm2(x)
        residual = x  # Save for next residual
        x = self.dropout2(x)
        x = self.activation2(x)

        # Layer 3 with residual
        x = self.fc3(x)
        x = x + residual  # Residual connection

        return x


class IdentityProjectionHead(nn.Module):
    """Pass-through projection used when embeddings are already aligned."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        if input_dim != output_dim:
            raise ValueError(
                "Identity projection requires input_dim == output_dim; "
                f"got {input_dim} vs {output_dim}."
            )
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple passthrough
        return tokens


class ColAF_MLP_Wide(nn.Module):
    """
    Wider variant with larger hidden dimensions and output dimension.
    Good for maximum information preservation.

    Architecture:
        Input (3584) → Linear(3584→2048) → LayerNorm+Dropout→GELU
        → Linear(2048→1024) → LayerNorm+Dropout→GELU
        → Linear(1024→768)
    """

    def __init__(
        self,
        input_dim: int = 3584,
        hidden_dim1: int = 2048,
        hidden_dim2: int = 1024,
        output_dim: int = 768,
        dropout: float = 0.1,
        bias: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Layer 1: 3584 → 2048
        self.fc1 = nn.Linear(input_dim, hidden_dim1, bias=bias)
        self.norm1 = nn.LayerNorm(hidden_dim1)
        self.dropout1 = nn.Dropout(dropout)
        self.activation1 = nn.GELU()

        # Layer 2: 2048 → 1024
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2, bias=bias)
        self.norm2 = nn.LayerNorm(hidden_dim2)
        self.dropout2 = nn.Dropout(dropout)
        self.activation2 = nn.GELU()

        # Layer 3: 1024 → 768
        self.fc3 = nn.Linear(hidden_dim2, output_dim, bias=bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (..., input_dim)

        Returns:
            (..., output_dim) tensor after wide projection
        """
        x = self.fc1(tokens)
        x = self.norm1(x)
        x = self.dropout1(x)
        x = self.activation1(x)

        x = self.fc2(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        x = self.activation2(x)

        x = self.fc3(x)

        return x


def build_projection_head(
    input_dim: int,
    output_dim: int,
    bias: bool = False,
    dropout: float = 0.0,
    architecture: str = "baseline",
) -> nn.Module:
    """
    Factory to build different projection head architectures.

    Args:
        input_dim: Input dimension (typically 3584 for AudioFlamingo)
        output_dim: Output projection dimension
        bias: Whether to use bias in linear layers
        dropout: Dropout probability
        architecture: Which architecture to use:
            - "baseline": Original simple 2-layer projection
            - "mlp_deep": ColAF_MLP_Deep with residuals (recommended)
            - "mlp_wide": ColAF_MLP_Wide with larger dimensions
            - "identity": Pass-through head (requires input_dim == output_dim)

    Returns:
        Projection head module
    """
    if architecture == "identity":
        return IdentityProjectionHead(input_dim=input_dim, output_dim=output_dim)
    if architecture == "mlp_deep":
        return ColAF_MLP_Deep(
            input_dim=input_dim,
            output_dim=output_dim,
            dropout=dropout,
            bias=bias,
        )
    elif architecture == "mlp_wide":
        return ColAF_MLP_Wide(
            input_dim=input_dim,
            output_dim=output_dim,
            dropout=dropout,
            bias=bias,
        )
    elif architecture == "baseline":
        config = ProjectionConfig(input_dim=input_dim, output_dim=output_dim, bias=bias, dropout=dropout)
        return TokenProjectionHead(config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}. Choose from: baseline, mlp_deep, mlp_wide")
