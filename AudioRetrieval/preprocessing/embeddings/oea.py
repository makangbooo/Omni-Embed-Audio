"""
OEA (Omni-Embed Audio) Embedding Precomputation.

Provides embedding precomputation using the Omni-Embed model with LoRA fine-tuning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from AudioRetrieval.preprocessing.embeddings.base import BaseEmbeddingPrecomputer


class OEAEmbeddingPrecomputer(BaseEmbeddingPrecomputer):
    """
    OEA (Omni-Embed Audio) embedding precomputer.

    Uses the Omni-Embed model with optional LoRA fine-tuning and projection heads.

    Args:
        checkpoint: Path to OEA checkpoint file
        repo_id: HuggingFace repository ID for base model
        local_path: Optional local path to model files
        cache_dir: Optional cache directory for model files
        torch_dtype: Optional torch dtype override
        device_map: Optional device map for model loading
        lora_rank: LoRA rank (default: 16)
        lora_alpha: LoRA alpha (default: 32)
        lora_dropout: LoRA dropout (default: 0.05)
        lora_targets: LoRA target modules
        projection_dim: Projection head dimension (default: 512)
        projection_dropout: Projection dropout (default: 0.1)
        passage_prefix: Prefix for passage encoding
        query_prefix: Prefix for query encoding
        device: Device for inference
        batch_size_audio: Batch size for audio encoding
        batch_size_text: Batch size for text encoding

    Example:
        >>> precomputer = OEAEmbeddingPrecomputer(
        ...     checkpoint="checkpoints/best.pt",
        ...     repo_id="nvidia/omni-embed-nemotron-3b",
        ...     device="cuda",
        ... )
        >>> audio_embeds = precomputer.encode_audio(["/path/to/audio.wav"])
        >>> text_embeds = precomputer.encode_text(["A dog barking"])
    """

    def __init__(
        self,
        checkpoint: str,
        repo_id: str = "nvidia/omni-embed-nemotron-3b",
        local_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        device_map: Optional[str] = None,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_targets: str = "q_proj,k_proj,v_proj,o_proj,qkv,out_proj",
        projection_dim: int = 512,
        projection_dropout: float = 0.1,
        passage_prefix: str = "passage:",
        query_prefix: str = "query:",
        device: str = "cuda",
        batch_size_audio: int = 16,
        batch_size_text: int = 128,
    ):
        super().__init__(device, batch_size_audio, batch_size_text)
        self.checkpoint = Path(checkpoint)
        self.repo_id = repo_id
        self.local_path = local_path
        self.cache_dir = cache_dir
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_targets = [t.strip() for t in lora_targets.split(",") if t.strip()]
        self.projection_dim = projection_dim
        self.projection_dropout = projection_dropout
        self.passage_prefix = passage_prefix
        self.query_prefix = query_prefix

        # These will be set during model loading
        self._adapter = None
        self._peft_model = None
        self._audio_head = None
        self._text_head = None

    def _load_model(self) -> Dict[str, Any]:
        """Load OEA adapter and checkpoint."""
        from AudioRetrieval.models.omni_embed_adapter import OmniEmbedAdapter

        # Import training module for LoRA attachment and projection heads
        try:
            from AudioRetrieval.training.oea import train_omniembed_lora as training_module
        except ImportError:
            # Fall back to scripts path
            script_path = Path(__file__).resolve().parents[3] / "scripts" / "training"
            if str(script_path) not in sys.path:
                sys.path.insert(0, str(script_path))
            import train_omniembed_lora_retrieval as training_module

        # Build adapter
        self._adapter = OmniEmbedAdapter(
            repo_id=self.repo_id,
            local_path=self.local_path,
            device=self.device,
            passage_prefix=self.passage_prefix,
            query_prefix=self.query_prefix,
        )

        # Create config namespace for LoRA
        cfg = argparse.Namespace(
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            lora_targets=self.lora_targets,
        )

        # Attach LoRA
        base_model = self._adapter.get_underlying_model()
        self._peft_model = training_module.attach_lora(base_model, cfg)
        self._adapter.set_underlying_model(self._peft_model)

        # Load checkpoint
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        checkpoint_data = torch.load(self.checkpoint, map_location=device)

        # Load LoRA weights
        self._peft_model.load_state_dict(checkpoint_data["lora_state_dict"], strict=False)

        # Build and load projection heads
        hidden_size = self._peft_model.config.text_config.hidden_size

        self._audio_head = training_module.ProjectionHead(
            hidden_size,
            self.projection_dim,
            self.projection_dropout,
        ).to(device)
        self._audio_head.load_state_dict(checkpoint_data["audio_head"])
        self._audio_head.eval()

        self._text_head = training_module.ProjectionHead(
            hidden_size,
            self.projection_dim,
            self.projection_dropout,
        ).to(device)
        self._text_head.load_state_dict(checkpoint_data["text_head"])
        self._text_head.eval()

        return {
            "adapter": self._adapter,
            "peft_model": self._peft_model,
            "audio_head": self._audio_head,
            "text_head": self._text_head,
        }

    def _encode_audio_batch(self, audio_paths: List[str]) -> np.ndarray:
        """Encode a batch of audio files."""
        if self._adapter is None:
            self._load_model()

        with torch.inference_mode():
            raw_embeddings = self._adapter.encode_audio(
                audio_paths,
                batch_size=self.batch_size_audio,
            )
            raw_tensor = torch.from_numpy(raw_embeddings).to(self.device)
            projected = self._audio_head(raw_tensor).cpu().numpy()

        return projected

    def _encode_text_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of text strings."""
        if self._adapter is None:
            self._load_model()

        with torch.inference_mode():
            raw_embeddings = self._adapter.encode_text(
                texts,
                batch_size=self.batch_size_text,
            )
            raw_tensor = torch.from_numpy(raw_embeddings).to(self.device)
            projected = self._text_head(raw_tensor).cpu().numpy()

        return projected

    def precompute_full_dataset(
        self,
        audio_dir: Path,
        captions_csv: Path,
        uiq_jsonl: Optional[Path],
        output_dir: Path,
        dataset_type: str = "clotho",
        split: str = "eval",
        uiq_categories: Optional[List[str]] = None,
    ) -> None:
        """
        Precompute all embeddings for a dataset.

        Args:
            audio_dir: Directory containing audio files
            captions_csv: Path to captions CSV file
            uiq_jsonl: Optional path to UIQ JSONL file
            output_dir: Output directory for embeddings
            dataset_type: Dataset type (clotho/audiocaps)
            split: Dataset split name
            uiq_categories: UIQ categories to compute (default: all standard)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        print(f"OEA Embedding Precomputation ({dataset_type})")
        print("=" * 60)
        print(f"Audio directory: {audio_dir}")
        print(f"Captions CSV: {captions_csv}")
        print(f"Checkpoint: {self.checkpoint}")
        print(f"Output directory: {output_dir}")
        print("=" * 60)

        # Load entries based on dataset type
        if dataset_type.lower() == "clotho":
            entries = self.load_clotho_entries(captions_csv, audio_dir, split)
        elif dataset_type.lower() == "audiocaps":
            entries = self.load_audiocaps_entries(captions_csv, audio_dir, split)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        print(f"Loaded {len(entries)} entries")

        # Precompute audio and caption embeddings
        self.precompute_dataset(entries, output_dir)

        # Precompute UIQ embeddings if provided
        if uiq_jsonl and uiq_jsonl.exists():
            if uiq_categories is None:
                uiq_categories = [
                    "Pragmatic",
                    "Subjective",
                    "Contextual",
                    "Negative/Contrastive",
                    "Cross-domain",
                ]

            uiq_data = self.load_uiq_queries(uiq_jsonl)
            valid_clip_ids = {e.clip_id for e in entries if e.audio_path}

            for category in uiq_categories:
                if category not in uiq_data:
                    print(f"Warning: UIQ category '{category}' not found")
                    continue

                category_queries = uiq_data[category]
                texts = []
                clip_ids = []
                for clip_id, query in sorted(category_queries.items()):
                    if clip_id in valid_clip_ids:
                        texts.append(query)
                        clip_ids.append(clip_id)

                if not texts:
                    print(f"Warning: No UIQ queries for '{category}'")
                    continue

                print(f"Computing UIQ embeddings for '{category}' ({len(texts)} queries)...")
                embeddings = self.encode_text(texts)

                bucket_safe = category.lower().replace("/", "_").replace("-", "_")
                output_path = output_dir / f"uiq_{bucket_safe}_embeddings.npz"
                np.savez_compressed(
                    output_path,
                    embeddings=embeddings,
                    texts=texts,
                    clip_ids=clip_ids,
                    bucket=category,
                )
                print(f"Saved to {output_path}")

        print()
        print("=" * 60)
        print("All embeddings precomputed successfully!")
        print("=" * 60)
