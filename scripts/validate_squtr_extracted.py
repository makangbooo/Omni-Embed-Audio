#!/usr/bin/env python3
"""Validate extracted SQuTR corpus/query/qrels/audio alignment and build a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator


AudioProbe = Callable[[Path], dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-root", type=Path, required=True)
    parser.add_argument("--structure-manifest", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--statistics-output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL line: {path}:{line_number}")
            try:
                value = json.loads(line, object_pairs_hook=strict_object)
            except Exception as exc:
                raise ValueError(f"invalid JSONL row: {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row: {path}:{line_number}")
            yield line_number, value


def required_id(row: dict[str, Any], path: Path, line_number: int) -> str:
    raw = row.get("_id", row.get("id"))
    value = str(raw) if raw is not None else ""
    if not value:
        raise ValueError(f"missing ID: {path}:{line_number}")
    return value


def load_corpus(path: Path) -> tuple[set[str], dict[str, int]]:
    ids: set[str] = set()
    rows = 0
    titled = 0
    for line_number, row in iter_jsonl(path):
        identifier = required_id(row, path, line_number)
        if identifier in ids:
            raise ValueError(f"duplicate corpus ID {identifier!r}: {path}")
        text = row.get("text", row.get("content", ""))
        title = row.get("title", "")
        if not isinstance(text, str) or not isinstance(title, str):
            raise ValueError(f"invalid corpus text/title types: {path}:{line_number}")
        if not text.strip() and not title.strip():
            raise ValueError(f"empty corpus document: {path}:{line_number}")
        ids.add(identifier)
        rows += 1
        titled += bool(title.strip())
    if not ids:
        raise ValueError(f"empty corpus: {path}")
    return ids, {"rows": rows, "titled_rows": titled}


def load_queries(path: Path, expected: int) -> dict[str, str]:
    queries: dict[str, str] = {}
    for line_number, row in iter_jsonl(path):
        identifier = required_id(row, path, line_number)
        text = row.get("text")
        if identifier in queries:
            raise ValueError(f"duplicate query ID {identifier!r}: {path}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"empty/invalid query text: {path}:{line_number}")
        queries[identifier] = text
    if len(queries) != expected:
        raise ValueError(f"query count mismatch: {path}: {len(queries)} != {expected}")
    return queries


def validate_qrels(
    path: Path, query_ids: set[str], corpus_ids: set[str]
) -> dict[str, Any]:
    pairs: set[tuple[str, str]] = set()
    qrel_query_ids: set[str] = set()
    scores: Counter[str] = Counter()
    for line_number, row in iter_jsonl(path):
        query_raw = row.get("query-id", row.get("query_id"))
        corpus_raw = row.get("corpus-id", row.get("corpus_id"))
        query_id = str(query_raw) if query_raw is not None else ""
        corpus_id = str(corpus_raw) if corpus_raw is not None else ""
        score = row.get("score")
        if (
            not query_id
            or not corpus_id
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
        ):
            raise ValueError(f"invalid qrels schema: {path}:{line_number}")
        if query_id not in query_ids:
            raise ValueError(f"qrels references unknown query {query_id!r}: {path}")
        if corpus_id not in corpus_ids:
            raise ValueError(f"qrels references unknown corpus ID {corpus_id!r}: {path}")
        pair = (query_id, corpus_id)
        if pair in pairs:
            raise ValueError(f"duplicate qrels pair {pair!r}: {path}")
        pairs.add(pair)
        qrel_query_ids.add(query_id)
        scores[str(score)] += 1
    missing = sorted(query_ids - qrel_query_ids)
    if missing:
        raise ValueError(f"queries without qrels: {path}: {missing[:20]}")
    return {
        "rows": len(pairs),
        "queries_with_qrels": len(qrel_query_ids),
        "query_coverage_exact": qrel_query_ids == query_ids,
        "score_counts": dict(sorted(scores.items())),
        "all_query_ids_resolved": True,
        "all_corpus_ids_resolved": True,
    }


def safe_audio_name(value: Any, path: Path, line_number: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing audio filename: {path}:{line_number}")
    candidate = PurePosixPath(value)
    if (
        candidate.is_absolute()
        or len(candidate.parts) != 1
        or candidate.name != value
        or candidate.suffix.casefold() != ".wav"
    ):
        raise ValueError(f"unsafe audio filename {value!r}: {path}:{line_number}")
    return value


def probe_audio(path: Path) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    with sf.SoundFile(str(path), mode="r") as stream:
        frames = int(stream.frames)
        if frames <= 0 or stream.samplerate <= 0 or stream.channels <= 0:
            raise ValueError(f"invalid audio stream metadata: {path}")
        first = stream.read(frames=1, dtype="float32", always_2d=True)
        stream.seek(frames - 1)
        last = stream.read(frames=1, dtype="float32", always_2d=True)
        if first.shape[0] != 1 or last.shape[0] != 1:
            raise ValueError(f"unable to decode first/last frame: {path}")
        if not bool(np.isfinite(first).all() and np.isfinite(last).all()):
            raise ValueError(f"non-finite first/last decoded frame: {path}")
        return {
            "sample_rate": int(stream.samplerate),
            "channels": int(stream.channels),
            "frames": frames,
            "duration_seconds": frames / float(stream.samplerate),
            "format": stream.format,
            "subtype": stream.subtype,
            "decode_ok": True,
            "decode_scope": "first_and_last_frame_plus_full_zip_member_crc",
        }


def finalize_manifest(candidate: Path, output: Path) -> tuple[str, str]:
    candidate_sha256 = sha256_file(candidate)
    if output.exists():
        existing_sha256 = sha256_file(output)
        if existing_sha256 != candidate_sha256:
            raise ValueError(
                f"existing manifest differs; refusing to overwrite: {output}; "
                f"candidate preserved at {candidate}"
            )
        candidate.unlink()
        return "verified_existing", existing_sha256
    candidate.replace(output)
    return "created", candidate_sha256


def validate_dataset(
    extract_root: Path,
    structure: dict[str, Any],
    manifest_output: Path,
    progress_every: int,
    audio_probe: AudioProbe = probe_audio,
) -> tuple[dict[str, Any], str, str]:
    archive_root = str(structure["observed_archive_root"])
    dataset_root = extract_root / archive_root
    if dataset_root.is_symlink() or not dataset_root.is_dir():
        raise ValueError(f"missing/unsafe extracted dataset root: {dataset_root}")

    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    candidate = manifest_output.with_name(
        f".{manifest_output.name}.candidate-{os.getpid()}"
    )
    if candidate.exists():
        raise ValueError(f"manifest candidate already exists: {candidate}")

    total_audio = 0
    total_duration = 0.0
    sample_rates: Counter[int] = Counter()
    channels: Counter[int] = Counter()
    subset_reports: list[dict[str, Any]] = []
    try:
        with candidate.open("x", encoding="utf-8", newline="\n") as manifest:
            for subset in structure["subsets"]:
                relative_path = str(subset["relative_path"])
                subset_root = dataset_root.joinpath(
                    *PurePosixPath(relative_path).parts
                )
                expected_queries = int(subset["expected_unique_queries"])
                corpus_ids, corpus_report = load_corpus(
                    subset_root / "corpus.jsonl"
                )
                queries = load_queries(
                    subset_root / "queries.jsonl",
                    expected_queries,
                )
                qrels_report = validate_qrels(
                    subset_root / "qrels/test.jsonl",
                    set(queries),
                    corpus_ids,
                )
                condition_reports: dict[str, Any] = {}

                for condition in structure["conditions"]:
                    condition_id = str(condition["id"])
                    metadata_path = (
                        subset_root / condition["documented_query_file"]
                    )
                    audio_root = subset_root / condition["audio_directory"]
                    if audio_root.is_symlink() or not audio_root.is_dir():
                        raise ValueError(f"missing/unsafe audio directory: {audio_root}")
                    metadata_ids: set[str] = set()
                    referenced_audio: set[str] = set()
                    text_mismatches = 0
                    text_mismatch_examples: list[str] = []
                    rows = 0

                    for line_number, row in iter_jsonl(metadata_path):
                        query_id = required_id(row, metadata_path, line_number)
                        if query_id in metadata_ids:
                            raise ValueError(
                                f"duplicate audio query ID {query_id!r}: {metadata_path}"
                            )
                        if query_id not in queries:
                            raise ValueError(
                                f"audio metadata references unknown query "
                                f"{query_id!r}: {metadata_path}"
                            )
                        text = row.get("text")
                        if not isinstance(text, str) or not text.strip():
                            raise ValueError(
                                f"invalid audio query text: "
                                f"{metadata_path}:{line_number}"
                            )
                        if text != queries[query_id]:
                            text_mismatches += 1
                            if len(text_mismatch_examples) < 20:
                                text_mismatch_examples.append(query_id)
                        audio_name = safe_audio_name(
                            row.get("audio"),
                            metadata_path,
                            line_number,
                        )
                        if audio_name in referenced_audio:
                            raise ValueError(
                                f"duplicate audio filename {audio_name!r}: "
                                f"{metadata_path}"
                            )
                        expected_snr = condition.get("snr_db")
                        if expected_snr is not None:
                            if row.get("snr_db") != expected_snr:
                                raise ValueError(
                                    f"SNR mismatch for {query_id!r}: "
                                    f"{row.get('snr_db')!r} != {expected_snr!r}"
                                )
                            noise_id = row.get("noise_id")
                            if not isinstance(noise_id, str) or not noise_id:
                                raise ValueError(
                                    f"missing noise_id for {query_id!r}: "
                                    f"{metadata_path}"
                                )
                        else:
                            noise_id = row.get("noise_id")

                        audio_path = audio_root / audio_name
                        if audio_path.is_symlink() or not audio_path.is_file():
                            raise ValueError(f"missing/unsafe audio file: {audio_path}")
                        decoded = audio_probe(audio_path)
                        total_audio += 1
                        total_duration += float(decoded["duration_seconds"])
                        sample_rates[int(decoded["sample_rate"])] += 1
                        channels[int(decoded["channels"])] += 1
                        rows += 1
                        metadata_ids.add(query_id)
                        referenced_audio.add(audio_name)

                        manifest.write(
                            json.dumps(
                                {
                                    "sample_id": (
                                        f"{relative_path}:{condition_id}:{query_id}"
                                    ),
                                    "query_id": query_id,
                                    "dataset": "SQuTR",
                                    "subset": relative_path,
                                    "language": subset["language"],
                                    "condition": condition_id,
                                    "snr_db": expected_snr,
                                    "noise_id": noise_id,
                                    "audio_path": str(audio_path),
                                    "audio_relpath": audio_path.relative_to(
                                        extract_root
                                    ).as_posix(),
                                    "caption": text,
                                    "original_query_text": queries[query_id],
                                    "query_text_exact_match": (
                                        text == queries[query_id]
                                    ),
                                    "split": "test",
                                    "duration_seconds": decoded[
                                        "duration_seconds"
                                    ],
                                    "sample_rate": decoded["sample_rate"],
                                    "channels": decoded["channels"],
                                    "frames": decoded["frames"],
                                    "format": decoded["format"],
                                    "subtype": decoded["subtype"],
                                    "file_exists": True,
                                    "decode_ok": decoded["decode_ok"],
                                    "decode_scope": decoded["decode_scope"],
                                    "include_in_training": False,
                                    "filter_reason": "evaluation_only",
                                    "leakage_blocklist_status": (
                                        "NOT_APPLICABLE_EVALUATION_SPLIT"
                                    ),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        if total_audio % progress_every == 0:
                            print(
                                f"[PROGRESS] validated_audio={total_audio}/"
                                f"{structure['expected_audio_instances']}",
                                flush=True,
                            )

                    if metadata_ids != set(queries):
                        raise ValueError(
                            f"audio query ID set mismatch: {metadata_path}; "
                            f"metadata_only={sorted(metadata_ids-set(queries))[:20]}, "
                            f"query_only={sorted(set(queries)-metadata_ids)[:20]}"
                        )
                    actual_audio_names = {
                        path.name
                        for path in audio_root.iterdir()
                        if path.is_file() and path.suffix.casefold() == ".wav"
                    }
                    non_wav = sorted(
                        path.name
                        for path in audio_root.iterdir()
                        if path.is_file() and path.suffix.casefold() != ".wav"
                    )
                    nested_or_special = sorted(
                        path.name
                        for path in audio_root.iterdir()
                        if path.is_dir() or path.is_symlink()
                    )
                    if non_wav:
                        raise ValueError(
                            f"unexpected non-WAV files in {audio_root}: {non_wav[:20]}"
                        )
                    if nested_or_special:
                        raise ValueError(
                            f"unexpected nested/special entries in {audio_root}: "
                            f"{nested_or_special[:20]}"
                        )
                    if actual_audio_names != referenced_audio:
                        raise ValueError(
                            f"audio file/reference mismatch in {audio_root}: "
                            f"files_only={sorted(actual_audio_names-referenced_audio)[:20]}, "
                            f"metadata_only={sorted(referenced_audio-actual_audio_names)[:20]}"
                        )
                    if rows != expected_queries:
                        raise ValueError(
                            f"audio metadata count mismatch: {metadata_path}: "
                            f"{rows} != {expected_queries}"
                        )
                    condition_reports[condition_id] = {
                        "rows": rows,
                        "query_id_set_exact": True,
                        "audio_filename_set_exact": True,
                        "query_text_exact_matches": rows - text_mismatches,
                        "query_text_mismatches": text_mismatches,
                        "query_text_mismatch_examples": text_mismatch_examples,
                        "expected_snr_db": expected_snr,
                    }

                subset_reports.append(
                    {
                        "relative_path": relative_path,
                        "language": subset["language"],
                        "corpus": corpus_report,
                        "queries": len(queries),
                        "qrels": qrels_report,
                        "conditions": condition_reports,
                        "document_unit": "[CODE] one corpus.jsonl row",
                        "document_text_construction": (
                            "[CODE] title + newline + text when title is nonempty; "
                            "otherwise text"
                        ),
                    }
                )

        expected_audio = int(structure["expected_audio_instances"])
        if total_audio != expected_audio:
            raise ValueError(
                f"total validated audio mismatch: {total_audio} != {expected_audio}"
            )
        manifest_status, manifest_sha256 = finalize_manifest(
            candidate,
            manifest_output,
        )
        return (
            {
                "dataset": "SQuTR",
                "archive_root": archive_root,
                "subsets": subset_reports,
                "validated_audio_files": total_audio,
                "expected_audio_instances": expected_audio,
                "total_duration_seconds": total_duration,
                "total_duration_hours": total_duration / 3600.0,
                "sample_rate_counts": dict(sorted(sample_rates.items())),
                "channel_counts": dict(sorted(channels.items())),
                "audio_validation": (
                    "first and last frame decoded with libsndfile after every ZIP "
                    "member passed full CRC during extraction"
                ),
                "manifest_status": manifest_status,
                "manifest_path": str(manifest_output),
                "manifest_size_bytes": manifest_output.stat().st_size,
                "manifest_sha256": manifest_sha256,
                "source_tags": {
                    "dataset_structure": "[OBSERVED] DATA-13A",
                    "schema": "[CODE] official README and pinned omni_emb.py",
                    "document_unit": "[CODE] official CustomAudioRetrieval loader",
                    "query_text_differences": (
                        "[OBSERVED] reported without forcing equality because the "
                        "paper describes query text normalization"
                    ),
                },
            },
            manifest_status,
            manifest_sha256,
        )
    except Exception:
        if candidate.exists():
            print(f"[ERROR] manifest candidate preserved: {candidate}", flush=True)
        raise


def main() -> int:
    args = parse_args()
    extract_root = args.extract_root.expanduser().absolute()
    if extract_root.is_symlink():
        raise ValueError(f"extract root must not be a symbolic link: {extract_root}")
    structure = read_json(args.structure_manifest)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at": utc_now(),
        "extract_root": str(extract_root),
        "structure_manifest": str(args.structure_manifest.resolve()),
        "manifest_output": str(args.manifest_output.resolve()),
    }
    write_json(args.statistics_output, report)
    try:
        validation, _, _ = validate_dataset(
            extract_root,
            structure,
            args.manifest_output.resolve(),
            args.progress_every,
        )
        report.update(
            {
                "status": "complete",
                "finished_at": utc_now(),
                **validation,
            }
        )
        write_json(args.statistics_output, report)
        return 0
    except Exception as exc:  # noqa: BLE001 - preserve failure evidence
        report["status"] = "failed"
        report["finished_at"] = utc_now()
        report["error"] = repr(exc)
        write_json(args.statistics_output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
