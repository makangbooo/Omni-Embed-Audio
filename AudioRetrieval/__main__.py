"""
Main entry point for python -m AudioRetrieval.

Usage:
    python -m AudioRetrieval --help
    python -m AudioRetrieval preprocess embeddings --help
    python -m AudioRetrieval train --help
    python -m AudioRetrieval generate-uiq --help
    python -m AudioRetrieval evaluate --help
"""

import sys
from AudioRetrieval.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
