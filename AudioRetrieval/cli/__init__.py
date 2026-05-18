"""
AudioRetrieval CLI Module.

Provides unified command-line interface for all AudioRetrieval operations.

Usage:
    python -m AudioRetrieval --help
    python -m AudioRetrieval preprocess --help
    python -m AudioRetrieval train --help
    python -m AudioRetrieval generate-uiq --help
    python -m AudioRetrieval evaluate --help
"""

from AudioRetrieval.cli.main import main

__all__ = ["main"]
