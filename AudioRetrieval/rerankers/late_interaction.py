"""Naive late-interaction reranker operating on precomputed token embeddings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from .base import BaseReranker, CandidateItem, RerankResult
from .optimizations import pool_tokens, stride_tokens, prune_query_tokens

LOGGER = logging.getLogger(__name__)


def _slugify(value: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in value.strip())
    sanitized = sanitized.strip("_")
    return sanitized or "value"


def _as_str(cell: np.ndarray) -> str:
    if cell.shape == ():
        return str(cell.item())
    if cell.size == 1:
        return str(cell.reshape(()).item())
    return str(cell)


@dataclass
class EmbeddingEntry:
    embeddings: np.ndarray
    mask: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None


class LateInteractionEmbeddingStore:
    """Lazy loader for precomputed AudioFlamingo embeddings."""

    def __init__(
        self,
        root: Path,
        audio_subdir: str = "audio",
        caption_subdir: str = "captions",
        caption_normalize: bool = True,
        filter_prefix_tokens: bool = True,
    ) -> None:
        self.root = Path(root)
        self.audio_dir = self.root / audio_subdir
        self.caption_dir = self.root / caption_subdir
        if not self.audio_dir.is_dir():
            raise FileNotFoundError(f"Audio embedding directory not found: {self.audio_dir}")
        self.caption_dir_available = self.caption_dir.is_dir()
        # When enabled, apply a more robust normalization (lowercase + collapsed whitespace)
        self.caption_normalize = caption_normalize
        self.filter_prefix_tokens = filter_prefix_tokens

        self._audio_index: Dict[str, Path] = {}
        self._caption_index: Dict[str, List[Path]] = {}
        self._metadata: Dict[str, Any] = {}
        self._caption_prefix_length: Optional[int] = None
        self._load_metadata()
        self._build_indices()

    def _load_metadata(self) -> None:
        """Load metadata.json to get prefix information."""
        metadata_path = self.root / "metadata.json"
        if metadata_path.exists():
            import json
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
                    LOGGER.info("Loaded metadata from %s", metadata_path)
            except Exception as exc:
                LOGGER.warning("Failed to load metadata from %s: %s", metadata_path, exc)
        else:
            LOGGER.warning("metadata.json not found at %s. Prefix filtering may not work correctly.", metadata_path)

    def _normalize_caption(self, text: str) -> str:
        # Collapse consecutive whitespace and lowercase; also strip leading/trailing space
        return " ".join(text.split()).strip().lower()

    def _build_indices(self) -> None:
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None

        # Build audio index
        audio_files = sorted(self.audio_dir.glob("*.npz"))
        LOGGER.info("Building audio index from %d files...", len(audio_files))

        iterator = tqdm(audio_files, desc="Loading audio embeddings", unit="file") if tqdm else audio_files
        for npz_path in iterator:
            clip_id = None
            audio_slug = None
            try:
                with np.load(npz_path, allow_pickle=False) as data:
                    if "clip_id" in data:
                        clip_id = _as_str(data["clip_id"])
                    if "audio_path" in data:
                        path_value = _as_str(data["audio_path"])
                        if path_value:
                            audio_slug = Path(path_value).stem
            except Exception as exc:  # pragma: no cover - malformed files
                LOGGER.warning("Failed to inspect %s: %s", npz_path, exc)
                continue
            if not clip_id:
                clip_id = npz_path.stem
            self._audio_index[clip_id] = npz_path
            if audio_slug:
                self._audio_index.setdefault(audio_slug, npz_path)
                combined = f"{clip_id}_{audio_slug}"
                self._audio_index.setdefault(combined, npz_path)

        LOGGER.info("✓ Built audio index with %d entries", len(self._audio_index))

        if not self.caption_dir_available:
            return

        # Build caption index
        caption_files = sorted(self.caption_dir.glob("*.npz"))
        LOGGER.info("Building caption index from %d files...", len(caption_files))

        iterator = tqdm(caption_files, desc="Loading caption embeddings", unit="file") if tqdm else caption_files
        for npz_path in iterator:
            caption_text = None
            try:
                with np.load(npz_path, allow_pickle=False) as data:
                    if "caption_text" in data:
                        caption_text = _as_str(data["caption_text"])
            except Exception as exc:  # pragma: no cover - malformed files
                LOGGER.warning("Failed to inspect caption %s: %s", npz_path, exc)
                continue
            if not caption_text:
                continue
            key = self._normalize_caption(caption_text) if self.caption_normalize else caption_text
            self._caption_index.setdefault(key, []).append(npz_path)

        LOGGER.info("✓ Built caption index with %d unique captions", len(self._caption_index))

    @property
    def has_captions(self) -> bool:
        return bool(self._caption_index)

    @lru_cache(maxsize=512)
    def load_audio(self, clip_id: str) -> EmbeddingEntry:
        # Try direct lookup first
        if clip_id in self._audio_index:
            npz_path = self._audio_index[clip_id]
        else:
            # Fallbacks try common alias formats used in retrieval outputs.
            parts = clip_id.split("_")
            # 1) audiocap_id prefix
            if len(parts) >= 1 and parts[0] in self._audio_index:
                npz_path = self._audio_index[parts[0]]
                LOGGER.debug("Resolved clip_id '%s' to audiocap_id '%s'", clip_id, parts[0])
            else:
                # 2) youtube_id_start suffix
                suffix = "_".join(parts[1:]) if len(parts) > 1 else ""
                if suffix and suffix in self._audio_index:
                    npz_path = self._audio_index[suffix]
                    LOGGER.debug("Resolved clip_id '%s' to audio slug '%s'", clip_id, suffix)
                else:
                    raise KeyError(f"No audio embedding found for clip_id='{clip_id}'")

        try:
            with np.load(npz_path, allow_pickle=False) as data:
                embeddings: Optional[np.ndarray] = None
                mask: Optional[np.ndarray] = None
                if "audio_features" in data:
                    embeddings = np.array(data["audio_features"], dtype=np.float32, copy=False)
                elif "audio_embed" in data:
                    # CLAP-style embeddings (single vector, not token-level)
                    embeddings = np.array(data["audio_embed"], dtype=np.float32, copy=False)
                    # Ensure 2D shape (num_tokens=1, dim) for consistency
                    if embeddings.ndim == 1:
                        embeddings = embeddings[np.newaxis, :]  # (dim,) -> (1, dim)
                elif "hidden_states" in data:
                    LOGGER.warning(
                        "audio_features missing in %s; falling back to hidden_states. "
                        "Late-interaction scores may be unreliable.",
                        npz_path,
                    )
                    embeddings = np.array(data["hidden_states"], dtype=np.float32, copy=False)
                else:  # pragma: no cover - invalid data
                    raise KeyError(f"'audio_features' not present in {npz_path}")

                if "audio_token_mask" in data:
                    mask = np.array(data["audio_token_mask"], dtype=np.float32, copy=False)
                elif "audio_feature_mask" in data:
                    mask = np.array(data["audio_feature_mask"], dtype=np.float32, copy=False)

                return EmbeddingEntry(embeddings=embeddings, mask=mask, metadata={"path": npz_path})
        except KeyError:
            raise
        except Exception as exc:  # pragma: no cover - data corruption
            LOGGER.warning(
                "Failed to load audio embedding for clip_id '%s' at %s: %s", clip_id, npz_path, exc
            )
            self._audio_index.pop(clip_id, None)
            raise KeyError(f"No audio embedding found for clip_id='{clip_id}'") from exc

    def _get_caption_prefix_length(self) -> int:
        """Calculate the number of tokens in caption_prefix (cached)."""
        if self._caption_prefix_length is not None:
            return self._caption_prefix_length

        caption_prefix = self._metadata.get("caption_prefix", "")
        if not caption_prefix or not self.filter_prefix_tokens:
            self._caption_prefix_length = 0
            return 0

        # Try to get input_ids from any caption file to determine prefix length
        if not self._caption_index:
            self._caption_prefix_length = 0
            return 0

        # Sample one caption file
        sample_path = next(iter(self._caption_index.values()))[0]
        try:
            with np.load(sample_path, allow_pickle=False) as data:
                if "input_ids" in data:
                    input_ids = data["input_ids"]
                    caption_text = _as_str(data["caption_text"]) if "caption_text" in data else ""

                    # Count prefix tokens by finding where caption content starts
                    # This is approximate: we count tokens until we see non-special tokens
                    # A more robust approach would tokenize the prefix separately
                    full_text = f"{caption_prefix}{caption_text}"

                    # Heuristic: prefix tokens are at the beginning
                    # We estimate by the ratio of prefix length to full text length
                    if len(full_text) > 0:
                        prefix_ratio = len(caption_prefix) / len(full_text)
                        prefix_token_count = int(len(input_ids) * prefix_ratio)
                        LOGGER.info(
                            "Estimated caption prefix length: %d tokens (prefix='%s')",
                            prefix_token_count,
                            caption_prefix[:50],
                        )
                        self._caption_prefix_length = prefix_token_count
                        return prefix_token_count
        except Exception as exc:
            LOGGER.warning("Failed to estimate caption prefix length: %s", exc)

        self._caption_prefix_length = 0
        return 0

    @lru_cache(maxsize=2048)
    def load_caption(self, caption_text: str) -> EmbeddingEntry:
        key = self._normalize_caption(caption_text) if self.caption_normalize else caption_text
        paths = self._caption_index.get(key)
        if not paths:
            raise KeyError(f"No caption embedding found for text='{caption_text[:80]}'")
        npz_path = paths[0]
        try:
            with np.load(npz_path, allow_pickle=False) as data:
                # Support both token-level (hidden_states) and sample-level (text_embed) embeddings
                if "hidden_states" in data:
                    embeddings = np.array(data["hidden_states"], dtype=np.float32, copy=False)
                elif "text_embed" in data:
                    # CLAP-style embeddings (single vector, not token-level)
                    embeddings = np.array(data["text_embed"], dtype=np.float32, copy=False)
                    # Ensure 2D shape (num_tokens=1, dim) for consistency
                    if embeddings.ndim == 1:
                        embeddings = embeddings[np.newaxis, :]  # (dim,) -> (1, dim)
                else:
                    raise KeyError(f"'hidden_states' or 'text_embed' not present in {npz_path}")

                mask = None
                if "attention_mask" in data:
                    mask = np.array(data["attention_mask"], dtype=np.float32, copy=False)
                elif "caption_token_mask" in data:
                    mask = np.array(data["caption_token_mask"], dtype=np.float32, copy=False)

                # Filter out prefix tokens if enabled
                if self.filter_prefix_tokens:
                    prefix_len = self._get_caption_prefix_length()
                    if prefix_len > 0 and embeddings.shape[0] > prefix_len:
                        embeddings = embeddings[prefix_len:]
                        if mask is not None:
                            mask = mask[prefix_len:]
                        LOGGER.debug("Filtered %d prefix tokens from caption embedding", prefix_len)

                return EmbeddingEntry(
                    embeddings=embeddings,
                    mask=mask,
                    metadata={
                        "path": npz_path,
                        "clip_id": _as_str(data["clip_id"]) if "clip_id" in data else None,
                    },
                )
        except Exception as exc:  # pragma: no cover - data corruption
            LOGGER.warning(
                "Failed to load caption embedding for text '%s' at %s: %s", caption_text[:80], npz_path, exc
            )
            self._caption_index.pop(key, None)
            raise KeyError(f"No caption embedding found for text='{caption_text[:80]}'") from exc


def _prepare_tokens(embeddings: np.ndarray, mask: Optional[np.ndarray], eps: float = 1e-6) -> np.ndarray:
    if embeddings.ndim != 2:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)
    tokens = embeddings.astype(np.float32, copy=False)
    if mask is not None:
        flat_mask = mask.reshape(-1) if mask.ndim > 1 else mask
        if flat_mask.shape[0] == tokens.shape[0]:
            valid = flat_mask > 0.0
            tokens = tokens[valid]
    if tokens.size == 0:
        return np.zeros((0, embeddings.shape[-1]), dtype=np.float32)
    norms = np.linalg.norm(tokens, axis=1, keepdims=True)
    valid_rows = norms.squeeze(-1) > eps
    tokens = tokens[valid_rows]
    if tokens.size == 0:
        return np.zeros((0, embeddings.shape[-1]), dtype=np.float32)
    norms = np.clip(np.linalg.norm(tokens, axis=1, keepdims=True), eps, None)
    return tokens / norms


def _late_interaction_score(
    query_tokens: np.ndarray,
    doc_tokens: np.ndarray,
    aggregation: str = "mean",
) -> float:
    if query_tokens.size == 0 or doc_tokens.size == 0:
        return float("-inf")
    sims = query_tokens @ doc_tokens.T  # [Q, D]
    per_query = sims.max(axis=1)
    if aggregation == "sum":
        return float(per_query.sum())
    if aggregation == "max":
        return float(per_query.max())
    # default mean
    return float(per_query.mean())


class LateInteractionClient:
    """Utility that computes late-interaction scores against precomputed embeddings."""

    def __init__(
        self,
        embedding_root: str,
        dataset: Optional[str] = None,
        model: Optional[str] = None,
        audio_subdir: str = "audio",
        caption_subdir: str = "captions",
        aggregation: str = "mean",
        blend_with_initial: float = 1.0,
        allow_missing_query: bool = False,
        query_source: str = "caption",
        candidate_source: str = "audio",
        filter_prefix_tokens: bool = True,
        projection_checkpoint: Optional[str] = None,
        projection_device: str = "cpu",
        audio_prefix_tokens: int = 0,
        text_prefix_tokens: int = 0,
        audio_max_tokens: Optional[int] = None,
        text_max_tokens: Optional[int] = None,
        # Token optimization parameters (for fast first-stage retrieval)
        token_pool_factor: int = 1,
        audio_token_stride: int = 1,
        text_token_stride: int = 1,
        query_token_keep_ratio: float = 1.0,
        query_pruning_method: str = "norm",
    ) -> None:
        base = Path(embedding_root)
        if dataset:
            base = base / _slugify(dataset)
        if model:
            base = base / _slugify(model)
        self.store = LateInteractionEmbeddingStore(
            base, audio_subdir=audio_subdir, caption_subdir=caption_subdir, filter_prefix_tokens=filter_prefix_tokens
        )
        self.aggregation = aggregation
        self.blend = float(blend_with_initial)
        self.allow_missing_query = allow_missing_query
        self._last_details: Optional[Dict[str, Any]] = None
        valid_sources = {"audio", "caption"}
        if query_source not in valid_sources:
            raise ValueError(f"Unsupported query_source '{query_source}'. Expected one of {valid_sources}.")
        if candidate_source not in valid_sources:
            raise ValueError(f"Unsupported candidate_source '{candidate_source}'. Expected one of {valid_sources}.")
        if candidate_source == "caption" and not self.store.has_captions:
            raise ValueError("Caption embeddings are not available in the precomputed store.")
        self.query_source = query_source
        self.candidate_source = candidate_source
        self.audio_prefix_tokens = max(0, int(audio_prefix_tokens))
        self.text_prefix_tokens = max(0, int(text_prefix_tokens))
        self.audio_max_tokens = int(audio_max_tokens) if audio_max_tokens is not None else None
        self.text_max_tokens = int(text_max_tokens) if text_max_tokens is not None else None
        # Token optimization parameters
        self.token_pool_factor = max(1, int(token_pool_factor))
        self.audio_token_stride = max(1, int(audio_token_stride))
        self.text_token_stride = max(1, int(text_token_stride))
        self.query_token_keep_ratio = max(0.0, min(1.0, float(query_token_keep_ratio)))
        self.query_pruning_method = query_pruning_method
        self._projection_device = projection_device
        self._projection_audio_head = None
        self._projection_text_head = None
        if projection_checkpoint:
            self._load_projection_heads(projection_checkpoint, projection_device)
        # Log optimization settings if enabled
        if self.token_pool_factor > 1 or self.audio_token_stride > 1 or self.query_token_keep_ratio < 1.0:
            LOGGER.info(
                f"Token optimizations enabled: pool={self.token_pool_factor}, "
                f"audio_stride={self.audio_token_stride}, text_stride={self.text_token_stride}, "
                f"query_keep={self.query_token_keep_ratio:.1%}"
            )

    def _load_projection_heads(self, checkpoint_path: str, device: str) -> None:
        try:
            import torch
            from AudioRetrieval.training.models.projection_heads import build_projection_head
        except ImportError as exc:  # pragma: no cover - optional dependency at runtime
            raise RuntimeError("torch and training projection modules are required for projection checkpoints") from exc

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = ckpt.get("config", {})
        dropout = config.get("dropout", 0.0)
        architecture = config.get("architecture", "baseline")

        audio_state = ckpt.get("audio_head")
        text_state = ckpt.get("text_head")
        if audio_state is None or text_state is None:
            raise RuntimeError(f"Projection checkpoint {checkpoint_path} missing head weights")

        # Infer input/output dimensions from state dict
        # For baseline architecture, use "proj.weight", for mlp_deep/wide use "fc3.weight" (final layer)
        if architecture == "baseline":
            audio_in = audio_state["proj.weight"].shape[1]
            audio_out = audio_state["proj.weight"].shape[0]
            text_in = text_state["proj.weight"].shape[1]
            text_out = text_state["proj.weight"].shape[0]
        else:  # mlp_deep or mlp_wide
            # Input dim from first layer (fc1), output dim from last layer (fc3)
            audio_in = audio_state["fc1.weight"].shape[1]
            audio_out = audio_state["fc3.weight"].shape[0]
            text_in = text_state["fc1.weight"].shape[1]
            text_out = text_state["fc3.weight"].shape[0]

        LOGGER.info(f"Loading projection heads: {architecture} architecture, {audio_in}→{audio_out} dim")

        self._projection_audio_head = build_projection_head(audio_in, audio_out, dropout=dropout, architecture=architecture)
        self._projection_audio_head.load_state_dict(audio_state)
        self._projection_audio_head.eval().to(device)

        self._projection_text_head = build_projection_head(text_in, text_out, dropout=dropout, architecture=architecture)
        self._projection_text_head.load_state_dict(text_state)
        self._projection_text_head.eval().to(device)

    def _project_tokens(
        self,
        tokens: np.ndarray,
        mask: Optional[np.ndarray],
        kind: str,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        head = self._projection_audio_head if kind == "audio" else self._projection_text_head
        prefix = self.audio_prefix_tokens if kind == "audio" else self.text_prefix_tokens
        max_tokens = self.audio_max_tokens if kind == "audio" else self.text_max_tokens
        stride = self.audio_token_stride if kind == "audio" else self.text_token_stride

        tokens = np.asarray(tokens)
        mask_arr = np.asarray(mask) if mask is not None else None

        if prefix and tokens.shape[0] > 0:
            tokens = tokens[prefix:]
            if mask_arr is not None:
                mask_arr = mask_arr[prefix:]

        if max_tokens is not None and tokens.shape[0] > max_tokens:
            tokens = tokens[:max_tokens]
            if mask_arr is not None:
                mask_arr = mask_arr[:max_tokens]

        if mask_arr is not None:
            mask_arr = mask_arr.astype(np.float32, copy=False)

        if head is not None and tokens.size:
            try:
                import torch
            except ImportError as exc:  # pragma: no cover - optional dependency at runtime
                raise RuntimeError("torch is required to apply projection heads") from exc
            device = next(head.parameters()).device
            with torch.no_grad():
                tensor = torch.from_numpy(np.ascontiguousarray(tokens)).to(device=device, dtype=torch.float32)
                projected = head(tensor).cpu().numpy().astype(np.float32, copy=False)
            tokens = projected
        else:
            tokens = tokens.astype(np.float32, copy=False)

        # Apply token optimizations AFTER projection
        # 1. Token striding (audio-specific or text-specific)
        if stride > 1 and tokens.shape[0] > 0:
            tokens, mask_arr = stride_tokens(tokens, mask_arr, stride=stride)

        # 2. Token pooling (applied to both audio and text)
        if self.token_pool_factor > 1 and tokens.shape[0] > 0:
            tokens, mask_arr = pool_tokens(tokens, mask_arr, pool_factor=self.token_pool_factor)

        return tokens, mask_arr

    def encode_query(self, query: str) -> Optional[np.ndarray]:
        if self.query_source == "audio":
            return self.encode_query_audio(query)
        try:
            entry = self.store.load_caption(query)
        except KeyError:
            if not self.allow_missing_query:
                LOGGER.warning("No caption embedding available for query text '%s'", query[:80])
            return None
        tokens_arr, mask_arr = self._project_tokens(entry.embeddings, entry.mask, "text")
        tokens = _prepare_tokens(tokens_arr, mask_arr)
        if tokens.size == 0:
            LOGGER.warning("Caption embedding for query '%s' is empty after masking.", query[:80])
            return None

        # Apply query token pruning if enabled
        if self.query_token_keep_ratio < 1.0 and tokens.shape[0] > 1:
            tokens, _, _ = prune_query_tokens(
                tokens,
                mask=None,  # Already applied in _prepare_tokens
                keep_ratio=self.query_token_keep_ratio,
                method=self.query_pruning_method,
            )

        return tokens

    def encode_query_audio(self, clip_id: str) -> Optional[np.ndarray]:
        try:
            entry = self.store.load_audio(clip_id)
        except KeyError:
            if not self.allow_missing_query:
                LOGGER.warning("No audio embedding available for clip_id '%s'", clip_id)
            return None
        tokens_arr, mask_arr = self._project_tokens(entry.embeddings, entry.mask, "audio")
        tokens = _prepare_tokens(tokens_arr, mask_arr)
        if tokens.size == 0:
            LOGGER.warning("Audio embedding for clip_id '%s' is empty after masking.", clip_id)
            return None

        # Apply query token pruning if enabled
        if self.query_token_keep_ratio < 1.0 and tokens.shape[0] > 1:
            tokens, _, _ = prune_query_tokens(
                tokens,
                mask=None,  # Already applied in _prepare_tokens
                keep_ratio=self.query_token_keep_ratio,
                method=self.query_pruning_method,
            )

        return tokens

    def encode_audio(self, clip_id: str) -> Optional[np.ndarray]:
        try:
            entry = self.store.load_audio(clip_id)
        except KeyError:
            LOGGER.warning("No audio embedding cached for clip_id '%s'", clip_id)
            return None
        tokens_arr, mask_arr = self._project_tokens(entry.embeddings, entry.mask, "audio")
        tokens = _prepare_tokens(tokens_arr, mask_arr)
        if tokens.size == 0:
            LOGGER.warning("Audio embedding for clip_id '%s' is empty after masking.", clip_id)
            return None
        return tokens

    def encode_caption(self, caption_text: str) -> Optional[np.ndarray]:
        try:
            entry = self.store.load_caption(caption_text)
        except KeyError:
            LOGGER.warning("No caption embedding cached for text '%s'", caption_text[:80])
            return None
        tokens_arr, mask_arr = self._project_tokens(entry.embeddings, entry.mask, "text")
        tokens = _prepare_tokens(tokens_arr, mask_arr)
        if tokens.size == 0:
            LOGGER.warning("Caption embedding for text '%s' is empty after masking.", caption_text[:80])
            return None
        return tokens

    def score_candidates(
        self,
        query_tokens: np.ndarray,
        candidates: Iterable[CandidateItem],
    ) -> Tuple[List[Tuple[float, CandidateItem]], Dict[str, Any]]:
        diagnostics: Dict[str, Any] = {
            "aggregation": self.aggregation,
            "blend_with_initial": self.blend,
            "query_tokens": int(query_tokens.shape[0]),
            "candidate_stats": {},
            "projection": {
                "audio": bool(self._projection_audio_head is not None),
                "text": bool(self._projection_text_head is not None),
            },
        }
        scored: List[Tuple[float, CandidateItem]] = []
        for cand in candidates:
            info: Dict[str, Any] = {"initial_score": float(cand.initial_score)}
            if self.candidate_source == "audio":
                item_tokens = self.encode_audio(cand.clip_id)
            else:
                caption_text = cand.caption or cand.metadata.get("candidate_caption")
                if not caption_text:
                    LOGGER.warning(
                        "Candidate %s is missing caption text needed for late-interaction scoring.",
                        cand.clip_id,
                    )
                    item_tokens = None
                else:
                    item_tokens = self.encode_caption(caption_text)

            if item_tokens is None or item_tokens.size == 0 or query_tokens.size == 0:
                final_score = float(cand.initial_score)
                info["fallback"] = True
            else:
                li_score = _late_interaction_score(query_tokens, item_tokens, aggregation=self.aggregation)
                info["li_score"] = li_score
                if np.isfinite(li_score):
                    if self.blend >= 0.0 and self.blend <= 1.0:
                        final_score = (1.0 - self.blend) * float(cand.initial_score) + self.blend * li_score
                        info["final_score"] = final_score
                    else:
                        final_score = li_score
                else:
                    final_score = float(cand.initial_score)
                    info["fallback"] = True
            scored.append((final_score, cand))
            info["tokens"] = int(item_tokens.shape[0]) if isinstance(item_tokens, np.ndarray) else 0
            diagnostics["candidate_stats"][int(cand.candidate_idx)] = info
        self._last_details = diagnostics
        return scored, diagnostics

    def get_last_details(self) -> Optional[Dict[str, Any]]:
        return self._last_details


class LateInteractionReranker(BaseReranker):
    """Reranker that applies naive late-interaction scoring over precomputed embeddings."""

    def rerank(self, query: str, candidates: List[CandidateItem]) -> RerankResult:
        subset = self.limit_candidates(candidates)
        start = perf_counter()
        diagnostics: Dict[str, Any] = {
            "type": "late_interaction",
            "original_order": [cand.candidate_idx for cand in subset],
        }
        if not subset:
            latency = perf_counter() - start
            return RerankResult(
                candidates=[],
                latency_seconds=latency,
                gpu_cost_seconds=0.0,
                extra={"mode": "late_interaction", "diagnostics": diagnostics},
            )

        query_tokens = self.client.encode_query(query)
        if query_tokens is None or query_tokens.size == 0:
            diagnostics["skipped"] = "missing_query_embedding"
            latency = perf_counter() - start
            return RerankResult(
                candidates=subset,
                latency_seconds=latency,
                gpu_cost_seconds=0.0,
                extra={"mode": "late_interaction", "diagnostics": diagnostics},
            )

        scored, client_diag = self.client.score_candidates(query_tokens, subset)
        diagnostics.update(client_diag)
        scored.sort(key=lambda x: x[0], reverse=True)
        ordered = [cand for _, cand in scored]
        reranked_order = [cand.candidate_idx for cand in ordered]
        diagnostics["reranked_order"] = reranked_order
        diagnostics["final_scores"] = {int(cand.candidate_idx): float(score) for score, cand in scored}
        latency = perf_counter() - start
        return RerankResult(
            candidates=ordered,
            latency_seconds=latency,
            gpu_cost_seconds=0.0,
            extra={"mode": "late_interaction", "diagnostics": diagnostics},
        )
