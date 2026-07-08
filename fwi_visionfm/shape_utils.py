from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.datasets import discover_npz_samples, load_npz_sample
from fwi_visionfm.split_utils import load_split_paths


def infer_shapes_from_paths(paths: list[str | Path], *, max_checks: int | None = None) -> dict[str, Any]:
    if not paths:
        raise ValueError("no npz sample paths provided")
    if max_checks is not None and max_checks <= 0:
        raise ValueError("max_checks must be positive when provided")

    normalized = [Path(path) for path in paths]
    if max_checks is not None:
        normalized = normalized[:max_checks]

    records_shapes: set[tuple[int, ...]] = set()
    velocity_shapes: set[tuple[int, ...]] = set()
    checked = 0
    for path in normalized:
        sample = load_npz_sample(path)
        records_shapes.add(tuple(int(v) for v in sample.records.shape))
        velocity_shapes.add(tuple(int(v) for v in sample.velocity.shape))
        checked += 1

    summary: dict[str, Any] = {
        "sample_count": checked,
        "records_shape_set": [list(shape) for shape in sorted(records_shapes)],
        "velocity_shape_set": [list(shape) for shape in sorted(velocity_shapes)],
        "is_uniform_records_shape": len(records_shapes) == 1,
        "is_uniform_velocity_shape": len(velocity_shapes) == 1,
        "inferred_depth": None,
        "inferred_width": None,
    }
    if len(velocity_shapes) == 1:
        inferred_depth, inferred_width = next(iter(velocity_shapes))
        summary["inferred_depth"] = int(inferred_depth)
        summary["inferred_width"] = int(inferred_width)
    return summary


def infer_npz_dataset_shape(data_dir: str | Path, max_checks: int | None = None) -> dict[str, Any]:
    paths = discover_npz_samples(data_dir)
    if not paths:
        raise ValueError(f"no npz samples found in {data_dir}")
    summary = infer_shapes_from_paths(paths, max_checks=max_checks)
    summary["data_dir"] = str(Path(data_dir))
    return summary


def infer_split_manifest_shape(split_manifest: str | Path) -> dict[str, Any]:
    split_paths = load_split_paths(split_manifest)
    all_paths = [path for items in split_paths.values() for path in items]
    if not all_paths:
        raise ValueError(f"split manifest has no sample paths: {split_manifest}")
    summary = infer_shapes_from_paths(all_paths)
    summary["split_manifest"] = str(Path(split_manifest))
    return summary


def assert_requested_shape_matches(
    inferred_depth: int | None,
    inferred_width: int | None,
    requested_depth: int | None,
    requested_width: int | None,
) -> None:
    if requested_depth is None or requested_width is None:
        return
    if inferred_depth is None or inferred_width is None:
        raise ValueError("dataset shape could not be inferred because velocity shapes are not uniform")
    if int(inferred_depth) != int(requested_depth) or int(inferred_width) != int(requested_width):
        raise ValueError(
            f"Requested depth={requested_depth},width={requested_width} but dataset velocity shape is "
            f"{inferred_depth}x{inferred_width}. Use --auto-shape or correct --depth/--width."
        )
