"""
AudioRetrieval Package.

A comprehensive framework for audio-text retrieval research, featuring:

Modules:
--------
- preprocessing: Embedding precomputation and hard negative mining
- training: OEA LoRA training and data loaders
- uiq_generation: User intent query generation
- evaluation: Retrieval evaluation with multiple metrics
- models: Model adapters (LAION-CLAP, MGA-CLAP, OEA, etc.)
- rerankers: Second-stage reranking framework

CLI:
----
    python -m AudioRetrieval --help
    python -m AudioRetrieval preprocess embeddings --model laion_clap ...
    python -m AudioRetrieval train --config config.yaml
    python -m AudioRetrieval generate-uiq --backend gpt --dataset clotho
    python -m AudioRetrieval evaluate --mode baseline --model laion_clap

Example:
--------
    from AudioRetrieval.preprocessing import OEAEmbeddingPrecomputer
    from AudioRetrieval.evaluation import BaselineRunner
    from AudioRetrieval.uiq_generation import UIQGenerator, QueryType

    # Precompute embeddings
    precomputer = OEAEmbeddingPrecomputer(checkpoint="model.pt")
    precomputer.precompute_full_dataset(...)

    # Evaluate
    runner = BaselineRunner(model_name="laion_clap")
    results = runner.run(audio_dir=..., captions_csv=...)

    # Generate UIQ
    generator = UIQGenerator(backend="gpt")
    queries = generator.generate(captions, QueryType.QUESTION)
"""

__version__ = "1.0.0"

# Lazy imports for main modules
def __getattr__(name):
    """Lazy import modules to avoid loading heavy dependencies at import time."""
    if name == "preprocessing":
        from AudioRetrieval import preprocessing
        return preprocessing
    elif name == "training":
        from AudioRetrieval import training
        return training
    elif name == "uiq_generation":
        from AudioRetrieval import uiq_generation
        return uiq_generation
    elif name == "evaluation":
        from AudioRetrieval import evaluation
        return evaluation
    elif name == "models":
        from AudioRetrieval import models
        return models
    elif name == "rerankers":
        from AudioRetrieval import rerankers
        return rerankers
    elif name == "cli":
        from AudioRetrieval import cli
        return cli
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "preprocessing",
    "training",
    "uiq_generation",
    "evaluation",
    "models",
    "rerankers",
    "cli",
]
