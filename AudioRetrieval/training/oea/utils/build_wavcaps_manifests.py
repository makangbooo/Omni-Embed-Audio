#!/usr/bin/env python3
"""
Build deterministic WavCaps train/validation manifests for Omni-Embed training.

This script crawls the raw WavCaps metadata JSON files, resolves the audio files
under the provided audio root, and writes CSV manifests consumable by
`train_omniembed_lora_retrieval.py` with the new `--dataset wavcaps` option.

Validation entries are selected deterministically via a hash of the clip id so
that the split remains stable across runs and machines.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".aac", ".m4a", ".ogg", ".aif", ".aiff"}


@dataclass
class WavCapsEntry:
    clip_id: str
    caption: str
    audio_relpath: Path
    source: str
    duration: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create WavCaps train/val manifests.")
    parser.add_argument("--metadata-root", type=Path, default=Path("WavCaps/metadata"))
    parser.add_argument("--audio-root", type=Path, default=Path("WavCaps/audio"))
    parser.add_argument("--output-dir", type=Path, default=Path("WavCaps/manifests"))
    parser.add_argument("--train-name", type=str, default="wavcaps_train.csv")
    parser.add_argument("--val-name", type=str, default="wavcaps_val.csv")
    parser.add_argument("--val-ratio", type=float, default=0.01, help="Fraction of clips per source to reserve for validation.")
    parser.add_argument("--split-seed", type=int, default=1337, help="Seed used inside the deterministic hash for the split.")
    parser.add_argument("--min-duration", type=float, default=0.0, help="Drop clips shorter than this many seconds.")
    parser.add_argument("--max-duration", type=float, default=None, help="Drop clips longer than this (None keeps all).")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate manifests even if they already exist.")
    return parser.parse_args()


def _iter_metadata_files(metadata_root: Path) -> Iterable[Tuple[str, Path]]:
    for json_path in sorted(metadata_root.rglob("*.json")):
        if not json_path.is_file():
            continue
        source = json_path.parent.name
        yield source, json_path


def _build_source_index(audio_root: Path, source: str) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    base = audio_root / source / "audio"
    if not base.exists():
        raise FileNotFoundError(f"Audio directory missing for source '{source}': {base}")
    exact: Dict[str, Path] = {}
    folded: Dict[str, Path] = {}
    for path in base.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        rel = path.relative_to(audio_root)
        stem = path.stem
        # Prefer the shortest relative path when duplicates exist
        prev = exact.get(stem)
        if prev is None or len(rel.parts) < len(prev.parts):
            exact[stem] = rel
            folded[stem.casefold()] = rel
        else:
            folded.setdefault(stem.casefold(), rel)
    if not exact:
        raise RuntimeError(f"No audio files indexed for source '{source}' under {base}")
    return exact, folded


def _resolve_audio_path(
    audio_root: Path,
    cache: Dict[str, Tuple[Dict[str, Path], Dict[str, Path]]],
    source: str,
    clip_id: str,
) -> Optional[Path]:
    if source not in cache:
        cache[source] = _build_source_index(audio_root, source)
    exact_map, folded_map = cache[source]
    stem = Path(clip_id).stem
    rel = exact_map.get(stem)
    if rel is None:
        rel = folded_map.get(stem.casefold())
    if rel is None:
        return None
    return rel


def _hash_to_float(token: str) -> float:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(1 << 64)


def _normalize_relpath(candidate: str, audio_root: Path) -> Optional[Path]:
    if not candidate:
        return None
    cand = Path(candidate)
    if cand.is_absolute():
        try:
            cand = cand.relative_to(audio_root)
        except ValueError:
            parts = cand.parts
            if audio_root.name in parts:
                idx = parts.index(audio_root.name)
                cand = Path(*parts[idx + 1 :])
            else:
                return None
    parts = list(cand.parts)
    if parts and parts[0] == audio_root.name:
        cand = Path(*parts[1:])
        parts = list(cand.parts)
    if parts and parts[0] == "audio":
        cand = Path(*parts[1:])
    return cand


def _find_integrated_metadata(metadata_root: Path) -> Optional[Path]:
    patterns = ("wavcaps_*subset*.json", "wavcaps_*lt31*.json", "wavcaps_*.json")
    for pattern in patterns:
        for candidate in sorted(metadata_root.glob(pattern)):
            if candidate.is_file():
                return candidate
    return None


def _load_integrated_entries(args: argparse.Namespace, metadata_path: Path) -> List[WavCapsEntry]:
    audio_root = args.audio_root
    with metadata_path.open() as f:
        payload = json.load(f)
    records = payload.get("data") or []
    if not records:
        raise RuntimeError(f"No entries found inside integrated metadata: {metadata_path}")

    entries: List[WavCapsEntry] = []
    dropped_audio = 0
    dropped_duration = 0

    for record in records:
        caption = (record.get("caption") or "").strip()
        if not caption:
            continue
        duration = float(record.get("duration") or 0.0)
        if args.min_duration and duration < args.min_duration:
            dropped_duration += 1
            continue
        if args.max_duration is not None and duration > args.max_duration:
            dropped_duration += 1
            continue

        rel = None
        for key in ("relative_audio_path", "subset_audio_path", "audio_relpath"):
            rel = _normalize_relpath(record.get(key, ""), audio_root)
            if rel is None:
                continue
            full = audio_root / rel
            if full.exists():
                break
            rel = None
        if rel is None:
            dropped_audio += 1
            continue

        source = (record.get("source") or rel.parts[0] if rel.parts else "unknown") or "unknown"
        clip_id_raw = Path(str(record.get("id") or Path(rel).stem)).stem
        entries.append(
            WavCapsEntry(
                clip_id=f"{source}_{clip_id_raw}",
                caption=caption,
                audio_relpath=rel,
                source=source,
                duration=duration,
            )
        )

    if not entries:
        raise RuntimeError(f"Integrated metadata produced zero usable entries: {metadata_path}")
    if dropped_audio:
        print(f"[WARN] Integrated metadata skipped {dropped_audio} entries with missing audio files.")
    if dropped_duration:
        print(f"[INFO] Integrated metadata filtered {dropped_duration} entries outside duration bounds.")
    entries.sort(key=lambda e: (e.source, e.clip_id))
    return entries


def _load_entries_from_shards(args: argparse.Namespace) -> List[WavCapsEntry]:
    audio_root = args.audio_root
    metadata_root = args.metadata_root
    if not metadata_root.exists():
        raise FileNotFoundError(f"Metadata root not found: {metadata_root}")
    if not audio_root.exists():
        raise FileNotFoundError(f"Audio root not found: {audio_root}")

    cache: Dict[str, Tuple[Dict[str, Path], Dict[str, Path]]] = {}
    entries: List[WavCapsEntry] = []
    dropped_audio = 0
    dropped_duration = 0

    for source, json_path in _iter_metadata_files(metadata_root):
        data = json.loads(json_path.read_text())
        clips = data.get("data", [])
        for record in clips:
            caption = (record.get("caption") or "").strip()
            if not caption:
                continue
            duration = float(record.get("duration") or 0.0)
            if args.min_duration and duration < args.min_duration:
                dropped_duration += 1
                continue
            if args.max_duration is not None and duration > args.max_duration:
                dropped_duration += 1
                continue
            clip_id = Path(str(record.get("id", ""))).stem
            if not clip_id:
                continue
            rel = _resolve_audio_path(audio_root, cache, source, clip_id)
            if rel is None:
                dropped_audio += 1
                continue
            entries.append(
                WavCapsEntry(
                    clip_id=f"{source}_{clip_id}",
                    caption=caption,
                    audio_relpath=rel,
                    source=source,
                    duration=duration,
                )
            )

    if not entries:
        raise RuntimeError("No WavCaps entries constructed. Verify paths and metadata format.")

    if dropped_audio:
        print(f"[WARN] Skipped {dropped_audio} entries with missing audio files.")
    if dropped_duration:
        print(f"[INFO] Filtered {dropped_duration} entries outside duration bounds.")

    entries.sort(key=lambda e: (e.source, e.clip_id))
    return entries


def _split_entries(entries: List[WavCapsEntry], val_ratio: float, seed: int) -> Tuple[List[WavCapsEntry], List[WavCapsEntry]]:
    grouped: Dict[str, List[WavCapsEntry]] = defaultdict(list)
    for entry in entries:
        grouped[entry.source].append(entry)

    train: List[WavCapsEntry] = []
    val: List[WavCapsEntry] = []
    for source in sorted(grouped.keys()):
        items = grouped[source]
        source_train: List[WavCapsEntry] = []
        source_val: List[WavCapsEntry] = []
        for entry in items:
            token = f"{entry.clip_id}_{seed}"
            if _hash_to_float(token) < val_ratio:
                source_val.append(entry)
            else:
                source_train.append(entry)
        if not source_val and source_train:
            # Guarantee at least one validation example per source
            source_val.append(source_train.pop())
        train.extend(source_train)
        val.extend(source_val)
    return train, val


def _write_manifest(path: Path, rows: Iterable[WavCapsEntry]) -> None:
    fieldnames = ["clip_id", "audio_relpath", "caption", "source", "duration", "caption_index"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in rows:
            writer.writerow(
                {
                    "clip_id": entry.clip_id,
                    "audio_relpath": entry.audio_relpath.as_posix(),
                    "caption": entry.caption,
                    "source": entry.source,
                    "duration": f"{entry.duration:.6f}",
                    "caption_index": 0,
                }
            )


def main() -> int:
    args = parse_args()
    args.metadata_root = args.metadata_root.resolve()
    args.audio_root = args.audio_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_csv = output_dir / args.train_name
    val_csv = output_dir / args.val_name

    if not args.overwrite and train_csv.exists() and val_csv.exists():
        print(f"[INFO] Manifests already exist at {output_dir}. Use --overwrite to regenerate.")
        return 0

    integrated_path = _find_integrated_metadata(args.metadata_root)
    if integrated_path:
        print(f"[INFO] Found integrated metadata JSON: {integrated_path}")
        entries = _load_integrated_entries(args, integrated_path)
    else:
        print(f"[INFO] Loading WavCaps metadata from {args.metadata_root} ...")
        entries = _load_entries_from_shards(args)
    print(f"[INFO] Parsed {len(entries):,} caption/audio pairs across {len({e.source for e in entries})} sources.")

    train_entries, val_entries = _split_entries(entries, args.val_ratio, args.split_seed)
    print(f"[INFO] Split into {len(train_entries):,} train and {len(val_entries):,} val entries "
          f"({len(val_entries)/len(entries):.2%} val).")

    print(f"[INFO] Writing train manifest to {train_csv}")
    _write_manifest(train_csv, train_entries)
    print(f"[INFO] Writing val manifest to {val_csv}")
    _write_manifest(val_csv, val_entries)

    return 0


if __name__ == "__main__":
    sys.exit(main())
