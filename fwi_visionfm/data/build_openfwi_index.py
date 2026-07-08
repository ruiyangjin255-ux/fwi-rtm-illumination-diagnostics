from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


_DATA_PATTERN = re.compile(r"^data(?P<index>\d+)\.npy$", re.IGNORECASE)
_MODEL_PATTERN = re.compile(r"^model(?P<index>\d+)\.npy$", re.IGNORECASE)


def _family_root(path: Path) -> Path:
    parent = path.parent
    if parent.name.lower() in {"data", "model"}:
        return parent.parent
    return parent


def _family_name(path: Path) -> str:
    return _family_root(path).name


def _discover_pairs(openfwi_root: Path) -> dict[str, list[tuple[int, Path, Path]]]:
    data_files: dict[tuple[Path, int], Path] = {}
    model_files: dict[tuple[Path, int], Path] = {}
    for path in openfwi_root.rglob("*.npy"):
        data_match = _DATA_PATTERN.match(path.name)
        if data_match:
            data_files[(_family_root(path), int(data_match.group("index")))] = path
            continue
        model_match = _MODEL_PATTERN.match(path.name)
        if model_match:
            model_files[(_family_root(path), int(model_match.group("index")))] = path
    keys = sorted(set(data_files.keys()) & set(model_files.keys()), key=lambda item: (item[0].name.lower(), item[1]))
    if not keys:
        raise ValueError(f"No paired data/model files found under {openfwi_root}")
    grouped: dict[str, list[tuple[int, Path, Path]]] = defaultdict(list)
    for family_root, file_id in keys:
        grouped[family_root.name].append((file_id, data_files[(family_root, file_id)], model_files[(family_root, file_id)]))
    return grouped


def build_openfwi_index(
    *,
    openfwi_root: str | Path,
    output_path: str | Path,
    max_files_per_family: int | None = None,
) -> dict[str, Any]:
    root = Path(openfwi_root)
    grouped = _discover_pairs(root)
    fieldnames = [
        "family",
        "data_file",
        "model_file",
        "file_id",
        "local_index",
        "global_index",
        "num_shots",
        "nt",
        "num_receivers",
        "velocity_channels",
        "velocity_height",
        "velocity_width",
    ]
    rows: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    global_index = 0
    for family in sorted(grouped):
        pairs = grouped[family]
        if max_files_per_family is not None:
            pairs = pairs[: max_files_per_family]
        local_count = 0
        for file_id, data_path, model_path in pairs:
            data_array = np.load(data_path, mmap_mode="r")
            model_array = np.load(model_path, mmap_mode="r")
            if data_array.ndim != 4:
                raise ValueError(f"records file must be 4D, got {data_array.shape} at {data_path}")
            if model_array.ndim != 4 or model_array.shape[1] != 1:
                raise ValueError(f"model file must be [N,1,H,W], got {model_array.shape} at {model_path}")
            if data_array.shape[0] != model_array.shape[0]:
                raise ValueError(f"sample count mismatch: {data_path} vs {model_path}")
            for local_index in range(int(data_array.shape[0])):
                rows.append(
                    {
                        "family": family,
                        "data_file": str(data_path),
                        "model_file": str(model_path),
                        "file_id": int(file_id),
                        "local_index": int(local_index),
                        "global_index": int(global_index),
                        "num_shots": int(data_array.shape[1]),
                        "nt": int(data_array.shape[2]),
                        "num_receivers": int(data_array.shape[3]),
                        "velocity_channels": int(model_array.shape[1]),
                        "velocity_height": int(model_array.shape[2]),
                        "velocity_width": int(model_array.shape[3]),
                    }
                )
                global_index += 1
                local_count += 1
        family_counts[family] = local_count
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {
        "openfwi_root": str(root),
        "output": str(out),
        "sample_count": len(rows),
        "families": sorted(family_counts.keys()),
        "family_counts": family_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a sample-level OpenFWI manifest from raw data*.npy/model*.npy files.")
    parser.add_argument("--openfwi-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-files-per-family", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_openfwi_index(
        openfwi_root=args.openfwi_root,
        output_path=args.output,
        max_files_per_family=args.max_files_per_family,
    )
    print(f"写出 manifest: {args.output}")
    print(f"样本数: {summary['sample_count']}")
    print(f"families: {summary['families']}")


if __name__ == "__main__":
    main()
