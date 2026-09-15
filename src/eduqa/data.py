"""Validate source JSONL and produce reproducible, leakage-checked Alpaca datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
import unicodedata


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    """Normalize Unicode/newlines without flattening mathematical explanations."""
    return unicodedata.normalize("NFKC", value).replace("\r\n", "\n").replace("\r", "\n").strip()


def prompt_key(row: dict[str, str]) -> tuple[str, str]:
    return tuple(" ".join(normalize_text(row.get(field, "")).split()).casefold()
                 for field in ("instruction", "input"))


def load_jsonl(path: Path) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Reject malformed records and conflicting answers; report blank lines explicitly."""
    rows: list[dict[str, str]] = []
    seen: dict[tuple[str, str], tuple[str, int]] = {}
    counts = {"source_records": 0, "blank_lines": 0, "duplicates_removed": 0}
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                counts["blank_lines"] += 1
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(raw, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            unknown = set(raw) - {"instruction", "input", "output"}
            if unknown:
                raise ValueError(f"{path}:{line_number}: unsupported fields: {sorted(unknown)}")
            row: dict[str, str] = {}
            for field in ("instruction", "input", "output"):
                value = raw.get(field, "" if field == "input" else None)
                if not isinstance(value, str):
                    raise ValueError(f"{path}:{line_number}: {field} must be a string")
                row[field] = normalize_text(value)
                if field != "input" and not row[field]:
                    raise ValueError(f"{path}:{line_number}: {field} must not be empty")
            counts["source_records"] += 1
            key = prompt_key(row)
            if key in seen:
                answer, first_line = seen[key]
                if answer != row["output"]:
                    raise ValueError(
                        f"{path}:{line_number}: conflicting answers for the same prompt "
                        f"(first seen on line {first_line})"
                    )
                counts["duplicates_removed"] += 1
                continue
            seen[key] = (row["output"], line_number)
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: dataset contains no valid records")
    counts["unique_records"] = len(rows)
    return rows, counts


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_data(
    train_file: Path,
    test_file: Path,
    output_dir: Path,
    *,
    val_ratio: float = 0.1,
    seed: int = 42,
    overwrite: bool = False,
) -> dict:
    if not math.isfinite(val_ratio) or not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be strictly between 0 and 1")
    train_file, test_file, output_dir = train_file.resolve(), test_file.resolve(), output_dir.resolve()
    if output_dir in (train_file.parent, test_file.parent):
        raise ValueError("output_dir must be separate from source dataset directories")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())) and not overwrite:
        raise FileExistsError(f"{output_dir} already exists; use --overwrite explicitly")
    pool, train_counts = load_jsonl(train_file)
    test, test_counts = load_jsonl(test_file)
    overlap = {prompt_key(row) for row in pool} & {prompt_key(row) for row in test}
    if overlap:
        example = sorted(overlap)[0][0][:100]
        raise ValueError(f"Train/test leakage: {len(overlap)} normalized prompts overlap; example: {example!r}")
    if len(pool) < 2:
        raise ValueError("Training source needs at least 2 unique records for train/validation split")
    random.Random(seed).shuffle(pool)
    val_count = min(len(pool) - 1, max(1, round(len(pool) * val_ratio)))
    splits = {"train": pool[val_count:], "val": pool[:val_count], "test": test}
    info = {
        f"eduqa_{name}": {
            "file_name": f"{name}.json",
            "formatting": "alpaca",
            "columns": {"prompt": "instruction", "query": "input", "response": "output"},
        }
        for name in splits
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    # Complete validation and stage every file before touching an existing output.
    with tempfile.TemporaryDirectory(prefix=".eduqa-data-", dir=output_dir.parent) as temp:
        staging = Path(temp)
        for name, rows in splits.items():
            _write_json(staging / f"{name}.json", rows)
        _write_json(staging / "dataset_info.json", info)
        manifest = {
            "schema_version": 1,
            "seed": seed,
            "validation_ratio": val_ratio,
            "normalization": "NFKC + newline normalization + trim; prompt identity additionally collapses whitespace and casefolds",
            "sources": {
                "train": {"path": str(train_file), "sha256": sha256_file(train_file), **train_counts},
                "test": {"path": str(test_file), "sha256": sha256_file(test_file), **test_counts},
            },
            "counts": {name: len(rows) for name, rows in splits.items()},
            "train_test_prompt_overlap": 0,
            "split_prompt_overlap": 0,
            "files": {file.name: {"sha256": sha256_file(file)} for file in sorted(staging.glob("*.json"))},
            "scope": "Small educational demonstration dataset; no claim of production accuracy or pedagogical correctness.",
        }
        _write_json(staging / "manifest.json", manifest)
        output_dir.mkdir(parents=True, exist_ok=True)
        for file in staging.iterdir():
            file.replace(output_dir / file.name)
    return manifest


def _handle_prepare(args: argparse.Namespace) -> None:
    manifest = prepare_data(Path(args.train_file), Path(args.test_file), Path(args.output_dir),
                            val_ratio=args.val_ratio, seed=args.seed, overwrite=args.overwrite)
    print(json.dumps({"output_dir": str(Path(args.output_dir).resolve()), **manifest}, ensure_ascii=False, indent=2))


def add_parser(subparsers) -> None:
    parser = subparsers.add_parser("prepare-data", help="Validate JSONL, detect leakage and build Alpaca splits")
    parser.add_argument("--train-file", default="EDU-QA/data/science_ft_500.jsonl")
    parser.add_argument("--test-file", default="EDU-QA/data/test.jsonl")
    parser.add_argument("--output-dir", default="artifacts/data")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(func=_handle_prepare)
