"""
UIQ Text Embedding Precomputation.

Provides unified UIQ (User Intent Query) text embedding precomputation
supporting multiple model backends.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm


class UIQTextEmbeddingPrecomputer:
    """
    UIQ text embedding precomputer.

    Supports multiple backends: laion_clap, mga_clap, m2d_clap, oea,
    robust_clap.

    Args:
        model: Model backend to use
        device: Device for inference
        batch_size_text: Batch size for text encoding
        **model_kwargs: Additional model-specific arguments

    Example:
        >>> precomputer = UIQTextEmbeddingPrecomputer(
        ...     model="laion_clap",
        ...     device="cuda",
        ...     laion_ckpt="checkpoints/630k-audioset-best.pt",
        ... )
        >>> precomputer.precompute(
        ...     uiq_jsonl=Path("uiq_queries.jsonl"),
        ...     output_dir=Path("embeddings/"),
        ... )
    """

    SUPPORTED_MODELS = [
        "laion_clap",
        "mga_clap",
        "m2d_clap",
        "oea",
        "robust_clap",
    ]

    def __init__(
        self,
        model: str,
        device: str = "cuda",
        batch_size_text: int = 256,
        **model_kwargs,
    ):
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model}. Choose from {self.SUPPORTED_MODELS}")

        self.model_name = model
        self.device = device
        self.batch_size_text = batch_size_text
        self.model_kwargs = model_kwargs
        self._encode_fn: Optional[Callable[[List[str]], np.ndarray]] = None

    def _build_encoder(self) -> Callable[[List[str]], np.ndarray]:
        """Build and return the encoding function for the selected model."""
        if self.model_name == "laion_clap":
            from AudioRetrieval.models.laion_clap_adapter import LaionClapAdapter

            checkpoint = self.model_kwargs.get("laion_ckpt")
            tokenizer_paths = {
                "bert-base-uncased": self.model_kwargs.get(
                    "laion_bert_tokenizer"
                ),
                "roberta-base": self.model_kwargs.get(
                    "laion_roberta_tokenizer"
                ),
                "facebook/bart-base": self.model_kwargs.get(
                    "laion_bart_tokenizer"
                ),
            }
            if not checkpoint or not all(tokenizer_paths.values()):
                raise ValueError(
                    "LAION-CLAP requires 'laion_ckpt' and local BERT, "
                    "RoBERTa, and BART tokenizer paths"
                )
            adapter = LaionClapAdapter(
                ckpt_path=checkpoint,
                amodel=self.model_kwargs.get("laion_amodel", "HTSAT-tiny"),
                tmodel=self.model_kwargs.get("laion_tmodel", "roberta"),
                enable_fusion=False,
                tokenizer_paths=tokenizer_paths,
            )

            def _encode(batch: List[str]) -> np.ndarray:
                return adapter.encode_text(
                    batch,
                    batch_size=min(self.batch_size_text, len(batch)),
                    device=self.device,
                )

            return _encode

        elif self.model_name == "robust_clap":
            from AudioRetrieval.models.robust_clap_adapter import RobustClapAdapter

            adapter = RobustClapAdapter(
                ckpt_path=self.model_kwargs.get("robust_ckpt", "ModelCheckpoint/robust-clap/630k-audioset-best.pt"),
                amodel=self.model_kwargs.get("robust_amodel", "HTSAT-tiny"),
                tmodel=self.model_kwargs.get("robust_tmodel", "roberta"),
                enable_fusion=not self.model_kwargs.get("robust_disable_fusion", False),
                repo_root=self.model_kwargs.get("robust_repo", None),
                bert_tokenizer_path=self.model_kwargs.get(
                    "robust_bert_tokenizer"
                ),
                roberta_tokenizer_path=self.model_kwargs.get(
                    "robust_roberta_tokenizer"
                ),
                bart_tokenizer_path=self.model_kwargs.get(
                    "robust_bart_tokenizer"
                ),
            )

            def _encode(batch: List[str]) -> np.ndarray:
                return adapter.encode_text(
                    batch,
                    batch_size=min(self.batch_size_text, len(batch)),
                    device=self.device,
                )

            return _encode

        elif self.model_name == "mga_clap":
            from AudioRetrieval.models.mga_clap_adapter import MGAClapAdapter

            adapter = MGAClapAdapter(
                repo_path=self.model_kwargs.get("mga_repo", "AudioRetrieval/models/mga_clap"),
                ckpt_path=self.model_kwargs.get("mga_ckpt", "ModelCheckpoint/mga_clap/mga-clap.pt"),
                seconds=self.model_kwargs.get("mga_seconds", 10.0),
                device=self.device,
                bert_tokenizer_path=self.model_kwargs.get("mga_bert_tokenizer"),
                expected_checkpoint_sha256=self.model_kwargs.get(
                    "mga_checkpoint_sha256"
                ),
            )

            def _encode(batch: List[str]) -> np.ndarray:
                return adapter.encode_text(
                    batch,
                    batch_size=min(self.batch_size_text, len(batch)),
                    device=self.device,
                )

            return _encode

        elif self.model_name == "m2d_clap":
            from AudioRetrieval.models.m2d_clap_adapter import M2DClapAdapter

            checkpoint = self.model_kwargs.get("m2d_ckpt")
            tokenizer = self.model_kwargs.get("m2d_bert_tokenizer")
            if not checkpoint or not tokenizer:
                raise ValueError(
                    "M2D-CLAP requires 'm2d_ckpt' and "
                    "'m2d_bert_tokenizer' parameters"
                )
            adapter = M2DClapAdapter(
                weight_file=checkpoint,
                device=self.device,
                bert_tokenizer_path=tokenizer,
            )

            def _encode(batch: List[str]) -> np.ndarray:
                return adapter.encode_text(
                    batch,
                    batch_size=min(self.batch_size_text, len(batch)),
                    device=self.device,
                )

            return _encode

        elif self.model_name == "oea":
            # OEA requires checkpoint
            checkpoint = self.model_kwargs.get("oea_checkpoint")
            if not checkpoint:
                raise ValueError("OEA model requires 'oea_checkpoint' parameter")

            from AudioRetrieval.preprocessing.embeddings.oea import OEAEmbeddingPrecomputer

            oea_precomputer = OEAEmbeddingPrecomputer(
                checkpoint=checkpoint,
                repo_id=self.model_kwargs.get("oea_repo_id", "nvidia/omni-embed-nemotron-3b"),
                local_path=self.model_kwargs.get("oea_local_path"),
                device=self.device,
                batch_size_text=self.batch_size_text,
            )
            # Initialize model
            oea_precomputer._load_model()

            def _encode(batch: List[str]) -> np.ndarray:
                return oea_precomputer._encode_text_batch(batch)

            return _encode

        raise ValueError(f"Unknown model: {self.model_name}")

    @property
    def encode_fn(self) -> Callable[[List[str]], np.ndarray]:
        """Lazy-load and return the encoding function."""
        if self._encode_fn is None:
            self._encode_fn = self._build_encoder()
        return self._encode_fn

    def load_uiq_entries(
        self,
        jsonl_path: Path,
        override_type: Optional[str] = None,
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Load UIQ entries from JSONL file.

        Args:
            jsonl_path: Path to UIQ JSONL file
            override_type: Override query type for all entries

        Returns:
            Dictionary {query_type: [(clip_id, query), ...]}
        """
        data: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

        if not jsonl_path.exists():
            raise FileNotFoundError(f"UIQ JSONL not found: {jsonl_path}")

        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue

                item = json.loads(line)

                # Support multiple clip_id field names
                clip_id = (
                    item.get("audio_id")
                    or item.get("clip_id")
                    or item.get("target_audio")
                    or item.get("negative_audio")
                )
                if not clip_id:
                    continue

                # Support multiple query field names
                query = (
                    item.get("generated_query")
                    or item.get("query")
                    or item.get("imperative_query")
                    or item.get("negative_query")
                )
                if not query:
                    continue

                query_type = override_type or item.get("query_type") or "generic"
                query_type = str(query_type).strip().lower()

                data[query_type].append((str(clip_id).strip(), str(query).strip()))

        return data

    def encode_in_batches(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode texts in batches.

        Args:
            texts: List of text strings to encode
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings
        """
        outputs = []
        iterator = range(0, len(texts), max(1, self.batch_size_text))

        if show_progress:
            iterator = tqdm(iterator, desc="Encoding", unit="batch", leave=False)

        for idx in iterator:
            batch = texts[idx:idx + self.batch_size_text]
            outputs.append(self.encode_fn(batch))

        return np.concatenate(outputs, axis=0)

    def precompute(
        self,
        uiq_jsonl: Path,
        output_dir: Path,
        dataset: str = "unknown",
        query_type_override: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Precompute UIQ text embeddings.

        Args:
            uiq_jsonl: Path to UIQ JSONL file
            output_dir: Output directory for embeddings
            dataset: Dataset name for metadata
            query_type_override: Override query type for all entries

        Returns:
            Summary dictionary with query types and file paths
        """
        uiq_entries = self.load_uiq_entries(uiq_jsonl, query_type_override)
        if not uiq_entries:
            raise RuntimeError(f"No UIQ queries found in {uiq_jsonl}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary: Dict[str, Dict[str, Any]] = {}

        for query_type, pairs in sorted(uiq_entries.items()):
            clip_ids = [clip_id for clip_id, _ in pairs]
            texts = [text for _, text in pairs]

            print(f"[INFO] Encoding {len(texts)} '{query_type}' queries...")
            embeddings = self.encode_in_batches(texts)

            bucket_slug = query_type.replace("/", "_").replace(" ", "_")
            npz_path = output_dir / f"uiq_{bucket_slug}_embeddings.npz"

            np.savez_compressed(
                npz_path,
                embeddings=embeddings.astype(np.float32, copy=False),
                texts=np.array(texts, dtype=object),
                clip_ids=np.array(clip_ids, dtype=object),
                query_type=query_type,
            )

            print(f"[INFO] Saved embeddings to {npz_path}")
            summary[query_type] = {
                "num_queries": len(texts),
                "output_file": str(npz_path),
            }

        # Write metadata
        metadata = {
            "dataset": dataset,
            "uiq_jsonl": str(uiq_jsonl),
            "model": self.model_name,
            "query_types": summary,
        }
        metadata_path = output_dir / "metadata_uiq.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))
        print(f"[INFO] Metadata written to {metadata_path}")

        return summary
