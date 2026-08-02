#!/usr/bin/env python3
"""Validate the pinned archive-extracted Robust-CLAP source tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_SOURCE_REVISION = "d08d0e3c545fa22df0930fc0d090741aaa9e2cc1"
EXPECTED_FILES = {
    "src/laion_clap/hook.py": (
        9984,
        "f84ac89e86e18c9e7eaf02595c9d78d5549e2d305220723fde3f98e437f5d420",
    ),
    "src/laion_clap/clap_module/factory.py": (
        11030,
        "14eee4ebf25171a3c6c2912a3eb977a2086c2be3d775dc1d70dbfd1e14113ecf",
    ),
    "src/laion_clap/clap_module/model.py": (
        33776,
        "5da5fbc2bf7acd516814f92b628e1e2711fd0fa4d4ea41d4bb5589bc45727850",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source(source_dir: Path) -> dict[str, object]:
    source = source_dir.expanduser().resolve()
    marker = source / ".source_revision"
    revision = marker.read_text(encoding="utf-8").strip()
    if revision != EXPECTED_SOURCE_REVISION:
        raise ValueError(f"Robust-CLAP source revision marker mismatch: {revision}")

    files = {}
    for relative, (expected_bytes, expected_sha256) in EXPECTED_FILES.items():
        path = source / relative
        size_bytes = path.stat().st_size
        sha256 = sha256_file(path)
        if size_bytes != expected_bytes or sha256 != expected_sha256:
            raise ValueError(f"Robust-CLAP source identity mismatch: {relative}")
        files[relative] = {
            "size_bytes": size_bytes,
            "sha256": sha256,
        }
    return {
        "schema_version": 1,
        "status": "complete",
        "source_dir": str(source),
        "revision": revision,
        "revision_evidence": ".source_revision written after pinned codeload extraction",
        "files": files,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_source(args.source_dir)
    if args.output:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite source identity: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print("ROBUST_CLAP_SOURCE_VALIDATION_STATUS=complete")
    print(f"SOURCE_REVISION={report['revision']}")
    if args.output:
        print(f"OUTPUT={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
