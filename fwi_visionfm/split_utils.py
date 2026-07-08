from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.datasets import discover_npz_samples, load_npz_sample, split_sample_paths


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def _resolve_sample_path(data_dir: Path, sample_path: str) -> Path:
    candidate = Path(sample_path)
    if candidate.is_absolute():
        return candidate
    return (data_dir / candidate).resolve()


def load_dataset_manifest(data_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(data_dir) / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"缺少 manifest.json: {manifest_path}")
    manifest = read_json(manifest_path)
    manifest["_manifest_path"] = str(manifest_path)
    return manifest


def collect_dataset_samples(data_dirs: list[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for data_dir_raw in data_dirs:
        data_dir = Path(data_dir_raw)
        manifest = load_dataset_manifest(data_dir)
        manifest_samples = manifest.get("samples", [])
        if manifest_samples:
            for sample in manifest_samples:
                sample_path = _resolve_sample_path(data_dir, str(sample["path"]))
                records.append(
                    {
                        "path": str(sample_path),
                        "family": str(sample.get("family", manifest.get("family", ""))),
                        "split_name": str(sample.get("split_name", manifest.get("split_name", ""))),
                        "subset_name": str(manifest.get("subset_name", "")),
                        "dataset_name": str(manifest.get("dataset_name", "")),
                    }
                )
        else:
            for sample_path in discover_npz_samples(data_dir):
                records.append(
                    {
                        "path": str(sample_path.resolve()),
                        "family": str(manifest.get("family", "")),
                        "split_name": str(manifest.get("split_name", "")),
                        "subset_name": str(manifest.get("subset_name", "")),
                        "dataset_name": str(manifest.get("dataset_name", "")),
                    }
                )
    return records


def build_family_summary(sample_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for record in sample_records:
        family = record.get("family", "") or "unknown"
        bucket = summary.setdefault(family, {"total": 0, "datasets": set(), "subsets": set()})
        bucket["total"] += 1
        if record.get("dataset_name"):
            bucket["datasets"].add(record["dataset_name"])
        if record.get("subset_name"):
            bucket["subsets"].add(record["subset_name"])
    for family, bucket in summary.items():
        bucket["datasets"] = sorted(bucket["datasets"])
        bucket["subsets"] = sorted(bucket["subsets"])
    return summary


def split_records(
    sample_records: list[dict[str, Any]],
    *,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    split = split_sample_paths(
        [record["path"] for record in sample_records],
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=seed,
    )
    record_by_path = {str(Path(record["path"]).resolve()): record for record in sample_records}
    return {
        name: [record_by_path[str(Path(path).resolve())] for path in paths]
        for name, paths in split.items()
    }


def materialize_split_payload(
    split_records_map: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    mode: str,
    train_fraction: float | None = None,
    val_fraction: float | None = None,
) -> dict[str, Any]:
    all_records = [item for items in split_records_map.values() for item in items]
    payload: dict[str, Any] = {
        "train": [record["path"] for record in split_records_map.get("train", [])],
        "val": [record["path"] for record in split_records_map.get("val", [])],
        "test": [record["path"] for record in split_records_map.get("test", [])],
        "families": build_family_summary(all_records),
        "seed": int(seed),
        "mode": mode,
    }
    if train_fraction is not None:
        payload["train_fraction"] = float(train_fraction)
    if val_fraction is not None:
        payload["val_fraction"] = float(val_fraction)
    return payload


def load_split_paths(split_manifest: str | Path) -> dict[str, list[Path]]:
    payload = read_json(split_manifest)
    missing = [name for name in ("train", "val", "test") if name not in payload]
    if missing:
        raise ValueError(f"split manifest 缺少字段: {missing}")
    return {
        name: [Path(path) for path in payload.get(name, [])]
        for name in ("train", "val", "test")
    }


def validate_split_shapes(paths: list[str | Path], *, max_checks: int | None = None) -> dict[str, Any]:
    record_shapes: set[tuple[int, ...]] = set()
    velocity_shapes: set[tuple[int, ...]] = set()
    checked = 0
    for raw_path in paths:
        sample_path = Path(raw_path)
        sample = load_npz_sample(sample_path)
        record_shapes.add(tuple(int(v) for v in sample.records.shape))
        velocity_shapes.add(tuple(int(v) for v in sample.velocity.shape))
        checked += 1
        if max_checks is not None and checked >= max_checks:
            break
    return {
        "records_shape_set": [list(shape) for shape in sorted(record_shapes)],
        "velocity_shape_set": [list(shape) for shape in sorted(velocity_shapes)],
    }
