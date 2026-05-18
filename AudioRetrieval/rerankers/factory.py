"""Factory utilities to build rerankers from configuration dictionaries."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import BaseReranker
from .modes import (
    AudioListwiseReranker,
    AudioPointwiseReranker,
    CaptionListwiseReranker,
    CaptionPointwiseReranker,
)
from .mock_client import MockLALMClient
from .audio_flamingo_client import AudioFlamingoLALMClient
from .qwen_omni_client import Qwen3OmniLALMClient, QwenOmniLALMClient
from .qwen_caption_client import QwenCaptionLALMClient
from .text_embedding_caption_client import TextEmbeddingCaptionClient
from .late_interaction import LateInteractionClient, LateInteractionReranker
from .xacle_baseline_client import XACLEBaselineClient
from .diffatr_reranker import DiffATRClient, DiffATRReranker


MODE_REGISTRY = {
    "audio_native_pointwise": AudioPointwiseReranker,
    "audio_native_listwise": AudioListwiseReranker,
    "caption_proxy_pointwise": CaptionPointwiseReranker,
    "caption_proxy_listwise": CaptionListwiseReranker,
    "late_interaction": LateInteractionReranker,
    "diffatr": DiffATRReranker,
}

CLIENT_REGISTRY = {
    "mock": MockLALMClient,
    "qwen_omni": QwenOmniLALMClient,
    "qwen3_omni": Qwen3OmniLALMClient,
    "audio_flamingo": AudioFlamingoLALMClient,
    "text_embedding_caption": TextEmbeddingCaptionClient,
    "qwen_caption": QwenCaptionLALMClient,
    "late_interaction": LateInteractionClient,
    "xacle_baseline": XACLEBaselineClient,
    "diffatr": DiffATRClient,
}


def build_reranker(cfg: Optional[Any]) -> Optional[BaseReranker]:
    """
    Build a reranker based on a Hydra/OmegaConf-style configuration.

    Parameters
    ----------
    cfg:
        The ``reranker`` configuration section. May be ``None``.

    Returns
    -------
    BaseReranker or ``None`` if reranking is disabled.
    """

    if cfg is None:
        return None

    enabled = getattr(cfg, "enabled", None)
    if enabled is None:
        # Assume dict-like
        enabled = cfg.get("enabled", False)
    if not enabled:
        return None

    mode = cfg.get("mode", None)
    if not mode:
        raise ValueError("Reranker configuration requires a `mode` field.")
    if mode not in MODE_REGISTRY:
        raise ValueError(f"Unknown reranker mode '{mode}'. "
                         f"Available: {list(MODE_REGISTRY.keys())}")

    top_k = int(cfg.get("top_k", 50))

    client_cfg: Dict[str, Any] = cfg.get("client", {}) or {}
    client_name = client_cfg.get("name", "mock")
    if client_name not in CLIENT_REGISTRY:
        raise ValueError(f"Unknown LALM client '{client_name}'. "
                         f"Available: {list(CLIENT_REGISTRY.keys())}")

    client = CLIENT_REGISTRY[client_name](**client_cfg.get("params", {}))
    reranker_cls = MODE_REGISTRY[mode]

    name = cfg.get("name", f"{client_name}_{mode}")
    return reranker_cls(name=name, top_k=top_k, client=client)
