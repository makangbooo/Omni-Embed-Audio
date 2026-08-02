#!/usr/bin/env python3
"""Project the audited MECAT manifest to the vanilla short-all schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-candidates", type=int, default=848)
    parser.add_argument("--captions-per-candidate", type=int, default=3)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def project(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_manifest.resolve()
    output = args.output.resolve()
    if sha256_file(source) != args.source_sha256:
        raise ValueError("MECAT source manifest SHA256 mismatch")
    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != args.expected_candidates:
        raise ValueError("MECAT source candidate count mismatch")

    projected: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        caption_fields = row.get("caption_fields")
        if not isinstance(caption_fields, dict):
            raise TypeError(f"caption_fields is not an object for {sample_id}")
        raw_captions = caption_fields.get("short")
        if not isinstance(raw_captions, list):
            raise TypeError(f"short captions are not a list for {sample_id}")
        captions = [
            value.strip()
            for value in raw_captions
            if isinstance(value, str)
            and value.strip()
            and value.strip().casefold() != "none"
        ]
        if len(captions) != args.captions_per_candidate:
            raise ValueError(f"short caption count mismatch for {sample_id}")
        if row.get("file_exists") is not True or row.get("decode_ok") is not True:
            raise ValueError(f"audio validation is incomplete for {sample_id}")
        audio_path = Path(str(row.get("audio_path", ""))).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        if audio_path.stat().st_size != row.get("audio_size_bytes"):
            raise ValueError(f"audio size mismatch for {sample_id}")
        if sha256_file(audio_path) != row.get("audio_sha256"):
            raise ValueError(f"audio SHA256 mismatch for {sample_id}")
        projected.append(
            {
                "sample_id": sample_id,
                "audio_path": str(audio_path),
                "captions": captions,
                "file_exists": True,
                "decode_ok": True,
            }
        )

    sample_ids = [row["sample_id"] for row in projected]
    if sample_ids != sorted(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ValueError("MECAT sample IDs are not canonical and unique")
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in projected
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite different manifest: {output}")
    else:
        output.write_text(content, encoding="utf-8", newline="\n")
    return {
        "status": "complete",
        "candidate_count": len(projected),
        "caption_count": len(projected) * args.captions_per_candidate,
        "caption_field": "short",
        "output_sha256": sha256_file(output),
    }


def main() -> int:
    report = project(parse_args())
    print("VANILLA_MECAT_MANIFEST_STATUS=complete")
    print(f"CANDIDATE_COUNT={report['candidate_count']}")
    print(f"CAPTION_COUNT={report['caption_count']}")
    print(f"CAPTION_FIELD={report['caption_field']}")
    print(f"OUTPUT_SHA256={report['output_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
