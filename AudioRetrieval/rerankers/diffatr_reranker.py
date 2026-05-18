"""DiffATR-based reranker using diffusion model for audio-text retrieval."""

from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from .base import BaseReranker, CandidateItem, RerankResult

logger = logging.getLogger(__name__)


class DiffATRClient:
    """
    Client for DiffATR diffusion-based reranking.

    This client loads pre-trained projection heads and diffusion model,
    then uses DDIM sampling to generate retrieval scores.
    """

    def __init__(
        self,
        checkpoint_path: str,
        embedding_root: str,
        dataset: str = None,
        model: str = None,
        audio_subdir: str = "audio",
        caption_subdir: str = "captions",
        device: str = "cuda",
        diffusion_steps: int = 50,
        ddim_steps: Optional[int] = None,
        blend_with_initial: float = 0.5,
        use_contrastive_score: bool = True,
        query_source: str = "caption",
        candidate_source: str = "audio",
        # Optional weights for combining diffusion and contrastive scores
        diffusion_weight: Optional[float] = None,
        contrastive_weight: Optional[float] = None,
    ):
        """
        Initialize DiffATR client.

        Args:
            checkpoint_path: Path to diffatr_checkpoint.pt or diffusion_best.pt
            embedding_root: Root directory for precomputed embeddings
            dataset: Dataset subdirectory name
            model: Model subdirectory name
            audio_subdir: Audio embeddings subdirectory
            caption_subdir: Caption embeddings subdirectory
            device: Device to run inference on
            diffusion_steps: Number of diffusion steps used in training
            ddim_steps: Number of DDIM sampling steps (default: same as diffusion_steps)
            blend_with_initial: Weight for blending (0=only first-stage, 1=only diffusion)
            use_contrastive_score: Whether to use contrastive similarity in addition to diffusion
            query_source: Source for queries ("caption" or "audio")
            candidate_source: Source for candidates ("caption" or "audio")
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.diffusion_steps = diffusion_steps
        self.ddim_steps = ddim_steps if ddim_steps is not None else diffusion_steps
        self.blend_with_initial = float(blend_with_initial)
        self.use_contrastive_score = bool(use_contrastive_score)

        # Configure score blending weights (defaults to 0.7/0.3 if both present)
        if diffusion_weight is None and contrastive_weight is None:
            self.diffusion_weight = 0.7
            self.contrastive_weight = 0.3
        else:
            dw = 0.0 if diffusion_weight is None else float(diffusion_weight)
            cw = 0.0 if contrastive_weight is None else float(contrastive_weight)
            total = max(dw + cw, 1e-8)
            # Normalize to sum to 1 for stability
            self.diffusion_weight = dw / total
            self.contrastive_weight = cw / total

        # Validate and store query/candidate sources
        valid_sources = {"caption", "audio"}
        if query_source not in valid_sources:
            raise ValueError(f"Unsupported query_source '{query_source}'. Expected one of {valid_sources}.")
        if candidate_source not in valid_sources:
            raise ValueError(f"Unsupported candidate_source '{candidate_source}'. Expected one of {valid_sources}.")

        self.query_source = query_source
        self.candidate_source = candidate_source

        # Load checkpoint
        logger.info(f"Loading DiffATR checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.config = checkpoint.get("config", {})
        self.stage = checkpoint.get("stage", "unknown")

        # Build projection heads
        from AudioRetrieval.training.models import build_projection_head, l2_normalize

        self.l2_normalize = l2_normalize

        audio_input_dim = self.config.get("audio_input_dim", 3584)
        text_input_dim = self.config.get("text_input_dim", 3584)
        projection_dim = self.config.get("projection_dim", 512)
        dropout = self.config.get("dropout", 0.0)
        architecture = self.config.get("projection_architecture", "mlp_deep")

        self.audio_head = build_projection_head(
            audio_input_dim, projection_dim, dropout=dropout, architecture=architecture
        ).to(self.device)
        self.text_head = build_projection_head(
            text_input_dim, projection_dim, dropout=dropout, architecture=architecture
        ).to(self.device)

        # Load weights
        if "audio_head" not in checkpoint or "text_head" not in checkpoint:
            raise RuntimeError(f"Checkpoint missing projection head weights (stage={self.stage})")

        self.audio_head.load_state_dict(checkpoint["audio_head"])
        self.text_head.load_state_dict(checkpoint["text_head"])

        self.audio_head.eval()
        self.text_head.eval()

        # Load diffusion model if available
        self.diffusion_model = None
        if "diffusion_model" in checkpoint:
            from AudioRetrieval.training.models.diffusion import DiffusionRerankerModel

            diffusion_cfg = self.config.get("diffusion", {})
            diffusion_hidden_dim = diffusion_cfg.get("hidden_dim", 512)
            diffusion_mlp_dim = diffusion_cfg.get("mlp_dim", 1024)
            diffusion_dropout = diffusion_cfg.get("dropout", 0.1)

            self.diffusion_model = DiffusionRerankerModel(
                feature_dim=projection_dim,
                hidden_dim=diffusion_hidden_dim,
                mlp_dim=diffusion_mlp_dim,
                dropout=diffusion_dropout,
            ).to(self.device)

            self.diffusion_model.load_state_dict(checkpoint["diffusion_model"])
            self.diffusion_model.eval()

            logger.info(f"✓ Loaded diffusion model (stage={self.stage})")
        else:
            logger.warning(f"No diffusion model in checkpoint (stage={self.stage}). Will only use contrastive scoring.")

        # Build diffusion schedule
        if self.diffusion_model is not None:
            from AudioRetrieval.training.objectives.diffusion import build_diffusion_schedule

            diffusion_cfg = self.config.get("diffusion", {})
            beta_start = diffusion_cfg.get("beta_start", 1e-3)
            beta_end = diffusion_cfg.get("beta_end", 5e-2)
            schedule = diffusion_cfg.get("schedule", "cosine")

            _, _, alphas_cumprod = build_diffusion_schedule(
                num_steps=self.diffusion_steps,
                device=self.device,
                beta_start=beta_start,
                beta_end=beta_end,
                schedule=schedule,
            )
            self.alphas_cumprod = alphas_cumprod

        # Set up embedding store
        from AudioRetrieval.rerankers.late_interaction import LateInteractionEmbeddingStore

        embedding_base = Path(embedding_root)
        if dataset:
            embedding_base = embedding_base / dataset
        if model:
            embedding_base = embedding_base / model

        self.embedding_store = LateInteractionEmbeddingStore(
            embedding_base,
            audio_subdir=audio_subdir,
            caption_subdir=caption_subdir,
            filter_prefix_tokens=True,
        )

        logger.info(f"✓ DiffATR client initialized (device={self.device}, diffusion_steps={self.diffusion_steps})")

    def _prepare_embeddings(
        self,
        raw_embeddings: np.ndarray,
        mask: Optional[np.ndarray],
        modality: str,
    ) -> torch.Tensor:
        """
        Prepare raw embeddings: apply projection head and normalize.

        Args:
            raw_embeddings: (num_tokens, input_dim) raw features
            mask: Optional (num_tokens,) mask
            modality: "audio" or "text"

        Returns:
            (num_tokens, projection_dim) normalized projections
        """
        # Convert to tensor
        tokens = torch.from_numpy(raw_embeddings.astype(np.float32)).to(self.device)

        # Apply mask
        if mask is not None:
            mask_tensor = torch.from_numpy(mask.astype(np.float32)).to(self.device)
            if mask_tensor.dim() > 1:
                mask_tensor = mask_tensor.reshape(-1)
            valid = mask_tensor > 0
            tokens = tokens[valid]

        if tokens.size(0) == 0:
            return torch.zeros((0, self.config.get("projection_dim", 512)), device=self.device)

        # Apply projection head
        with torch.no_grad():
            if modality == "audio":
                projected = self.audio_head(tokens)
            else:
                projected = self.text_head(tokens)

            # Normalize token-level projections
            projected = self.l2_normalize(projected)

        return projected

    def _fuzzy_load_caption(self, query: str) -> Optional[Any]:
        """
        Try to find a caption embedding using fuzzy matching.

        This handles minor text variations between CSVs (e.g., "honks" vs "honk").

        Args:
            query: Query text that failed exact match

        Returns:
            EmbeddingEntry if fuzzy match found, None otherwise
        """
        # Normalize query
        query_norm = query.lower().strip()
        query_words = set(query_norm.split())

        # Search for best fuzzy match in caption index
        best_match = None
        best_score = 0

        for caption_key in self.embedding_store._caption_index.keys():
            caption_norm = caption_key.lower().strip()
            caption_words = set(caption_norm.split())

            # Compute Jaccard similarity
            intersection = len(query_words & caption_words)
            union = len(query_words | caption_words)
            if union > 0:
                score = intersection / union

                # Require high similarity (>= 0.85) to avoid false matches
                if score > best_score and score >= 0.85:
                    best_score = score
                    best_match = caption_key

        if best_match is not None:
            logger.info(f"Fuzzy matched query to caption (similarity={best_score:.3f})")
            logger.debug(f"  Query: {query[:80]}")
            logger.debug(f"  Match: {best_match[:80]}")
            try:
                return self.embedding_store.load_caption(best_match)
            except KeyError:
                pass

        return None

    def _ddim_sample(
        self,
        text_proj: torch.Tensor,
        audio_proj: torch.Tensor,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        DDIM sampling to generate retrieval scores.

        Args:
            text_proj: (num_queries, proj_dim) text projections (queries can be 1 at inference)
            audio_proj: (num_candidates, proj_dim) audio projections
            num_steps: Number of DDIM steps (default: self.ddim_steps)

        Returns:
            (num_queries, num_candidates) score matrix
        """
        if self.diffusion_model is None:
            raise RuntimeError("Diffusion model not loaded")

        if num_steps is None:
            num_steps = self.ddim_steps

        num_queries = text_proj.size(0)
        num_candidates = audio_proj.size(0)

        # Start from pure noise matching the (Q, N) matrix shape
        x_t = torch.randn(num_queries, num_candidates, device=self.device)

        # DDIM sampling schedule
        # Skip some steps for faster inference; clamp to avoid zero step size
        step_size = max(1, self.diffusion_steps // num_steps)
        timestep_seq = list(range(0, self.diffusion_steps, step_size))[::-1]

        with torch.no_grad():
            for i, t in enumerate(timestep_seq):
                # One timestep per query (rows)
                t_batch = torch.full((num_queries,), t, device=self.device, dtype=torch.long)

                # Predict x0 from xt
                pred_x0 = self.diffusion_model(text_proj, audio_proj, x_t, t_batch)

                if i < len(timestep_seq) - 1:
                    # DDIM update step
                    t_next = timestep_seq[i + 1]
                    alpha_t = self.alphas_cumprod[t]
                    alpha_t_next = self.alphas_cumprod[t_next]

                    # Predicted noise
                    pred_noise = (x_t - torch.sqrt(alpha_t) * pred_x0) / torch.sqrt(1 - alpha_t + 1e-8)

                    # Update
                    x_t = (
                        torch.sqrt(alpha_t_next) * pred_x0
                        + torch.sqrt(1 - alpha_t_next) * pred_noise
                    )
                else:
                    # Final step: return prediction
                    x_t = pred_x0

        return x_t

    def score_candidates(
        self,
        query: str,
        candidates: List[CandidateItem],
    ) -> tuple[List[tuple[float, CandidateItem]], Dict[str, Any]]:
        """
        Score candidates using DiffATR.

        Args:
            query: Query text
            candidates: List of candidate items

        Returns:
            List of (score, candidate) tuples and diagnostics dict
        """
        diagnostics = {
            "use_diffusion": self.diffusion_model is not None,
            "use_contrastive": self.use_contrastive_score,
            "blend_with_initial": self.blend_with_initial,
            "num_candidates": len(candidates),
        }

        # Load query embedding based on query_source
        try:
            if self.query_source == "caption":
                query_entry = self.embedding_store.load_caption(query)
                query_modality = "text"
            else:  # audio
                query_entry = self.embedding_store.load_audio(query)
                query_modality = "audio"
        except KeyError:
            # Try fuzzy matching as fallback (only for captions)
            if self.query_source == "caption":
                query_entry = self._fuzzy_load_caption(query)
                query_modality = "text"
            else:
                query_entry = None

            if query_entry is None:
                logger.warning(f"No {self.query_source} embedding for query: {query[:80]}")
                # Fall back to initial scores
                scored = [(float(cand.initial_score), cand) for cand in candidates]
                diagnostics["fallback"] = "missing_query"
                return scored, diagnostics
            else:
                diagnostics["fuzzy_matched"] = True

        # Prepare query
        query_proj = self._prepare_embeddings(
            query_entry.embeddings,
            query_entry.mask,
            query_modality
        )

        if query_proj.size(0) == 0:
            logger.warning("Empty query embedding after masking")
            scored = [(float(cand.initial_score), cand) for cand in candidates]
            diagnostics["fallback"] = "empty_query"
            return scored, diagnostics

        # Pool query to single vector (mean pooling)
        query_vec = query_proj.mean(dim=0, keepdim=True)  # (1, proj_dim)

        # Load and prepare all candidate embeddings based on candidate_source
        candidate_vecs = []
        valid_candidates = []

        for cand in candidates:
            try:
                # Load candidate embedding based on candidate_source
                if self.candidate_source == "audio":
                    cand_entry = self.embedding_store.load_audio(cand.clip_id)
                    cand_modality = "audio"
                else:  # caption
                    # For T2T, candidates are captions - use cand.caption if available
                    caption_text = getattr(cand, 'caption', None) or cand.clip_id
                    cand_entry = self.embedding_store.load_caption(caption_text)
                    cand_modality = "text"

                cand_proj = self._prepare_embeddings(
                    cand_entry.embeddings,
                    cand_entry.mask,
                    cand_modality
                )

                if cand_proj.size(0) > 0:
                    # Pool to single vector
                    cand_vec = cand_proj.mean(dim=0, keepdim=True)  # (1, proj_dim)
                    candidate_vecs.append(cand_vec)
                    valid_candidates.append(cand)
            except KeyError:
                logger.warning(f"No {self.candidate_source} embedding for {cand.clip_id}")
                continue

        if not candidate_vecs:
            logger.warning(f"No valid {self.candidate_source} embeddings found")
            scored = [(float(cand.initial_score), cand) for cand in candidates]
            diagnostics["fallback"] = f"no_valid_{self.candidate_source}"
            return scored, diagnostics

        # Stack into batch
        candidate_batch = torch.cat(candidate_vecs, dim=0)  # (num_valid, proj_dim)
        num_candidates = candidate_batch.size(0)

        # Compute scores
        scores = {}

        # Contrastive score (cosine similarity)
        if self.use_contrastive_score:
            with torch.no_grad():
                # Re-normalize pooled vectors to avoid scale drift
                qv = self.l2_normalize(query_vec)
                cb = self.l2_normalize(candidate_batch)
                # Efficient: single query vs all candidates
                contrastive_scores = (qv * cb).sum(dim=1)  # (num_valid,)
                scores["contrastive"] = contrastive_scores.cpu().numpy()

        # Diffusion score
        if self.diffusion_model is not None:
            # Efficient 1×N inference: one query vs N candidates
            # This matches the reranking scenario and is much faster than N×N
            # query_vec is already (1, proj_dim) from mean pooling above
            score_matrix = self._ddim_sample(query_vec, candidate_batch)  # (1, num_candidates)

            # Extract the single row of scores
            diffusion_scores = score_matrix[0]  # (num_candidates,)
            scores["diffusion"] = diffusion_scores.cpu().numpy()

        # Combine scores
        final_scores = np.zeros(len(valid_candidates), dtype=np.float32)

        has_diff = "diffusion" in scores
        has_cont = "contrastive" in scores

        # Normalize components into [0, 1] for blending
        diff_norm = None
        cont_norm = None

        if has_diff:
            d = scores["diffusion"].astype(np.float32)
            d_min, d_max = float(d.min()), float(d.max())
            if d_max - d_min < 1e-8:
                # Degenerate diffusion output; ignore it
                has_diff = False
            else:
                diff_norm = (d - d_min) / (d_max - d_min)

        if has_cont:
            c = scores["contrastive"].astype(np.float32)
            # Cosine in [-1,1] -> [0,1]
            cont_norm = (c + 1.0) * 0.5

        if has_diff and has_cont:
            final_scores = self.diffusion_weight * diff_norm + self.contrastive_weight * cont_norm
        elif has_diff:
            final_scores = diff_norm  # already [0,1]
        elif has_cont:
            final_scores = cont_norm
        else:
            # Should not happen; fallback to initial ordering
            final_scores = np.array([float(c.initial_score) for c in valid_candidates], dtype=np.float32)

        # Blend with initial scores
        scored = []
        for i, cand in enumerate(valid_candidates):
            rerank_score = float(final_scores[i])

            if self.blend_with_initial > 0 and self.blend_with_initial < 1:
                # Normalize initial score (assume it's cosine similarity in [-1, 1])
                initial_norm = (float(cand.initial_score) + 1) / 2
                blended_score = (1 - self.blend_with_initial) * initial_norm + self.blend_with_initial * rerank_score
            else:
                blended_score = rerank_score

            scored.append((blended_score, cand))

        diagnostics["num_valid"] = len(valid_candidates)
        diagnostics["num_fallback"] = len(candidates) - len(valid_candidates)

        return scored, diagnostics


class DiffATRReranker(BaseReranker):
    """Reranker using DiffATR diffusion model."""

    def __init__(self, name: str, top_k: int, client: DiffATRClient):
        # Store client directly without calling super().__init__
        # since BaseReranker expects LALMClient protocol
        self.name = name
        self.top_k = top_k
        self.client = client

    def limit_candidates(self, candidates: List[CandidateItem]) -> List[CandidateItem]:
        """Clamp the number of candidates according to top_k."""
        if self.top_k > 0 and len(candidates) > self.top_k:
            return candidates[: self.top_k]
        return candidates

    def rerank(self, query: str, candidates: List[CandidateItem]) -> RerankResult:
        """Rerank candidates using DiffATR."""
        subset = self.limit_candidates(candidates)
        start = perf_counter()

        if not subset:
            return RerankResult(
                candidates=[],
                latency_seconds=perf_counter() - start,
                gpu_cost_seconds=0.0,
                extra={"mode": "diffatr"},
            )

        scored, diagnostics = self.client.score_candidates(query, subset)

        # Sort by score (descending)
        scored.sort(key=lambda x: x[0], reverse=True)
        ordered = [cand for _, cand in scored]

        latency = perf_counter() - start

        return RerankResult(
            candidates=ordered,
            latency_seconds=latency,
            gpu_cost_seconds=latency,  # Assume all time is GPU time
            extra={
                "mode": "diffatr",
                "diagnostics": diagnostics,
                "scores": {cand.candidate_idx: score for score, cand in scored},
            },
        )
