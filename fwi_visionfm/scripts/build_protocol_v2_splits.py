from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_DATA_PATTERN = re.compile(r"^data(?P<index>\d+)\.npy$", re.IGNORECASE)
_MODEL_PATTERN = re.compile(r"^model(?P<index>\d+)\.npy$", re.IGNORECASE)
_SPLIT_FIELDS = ["family", "data_file", "model_file", "local_index", "global_index", "path"]


def resolve_data_root(explicit: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit))
    if os.environ.get("OPENFWI_ROOT"):
        candidates.append(Path(os.environ["OPENFWI_ROOT"]))
    candidates.extend(
        [
            Path(r"D:\ryjin\OpenFWI"),
            Path(r"D:\ryjin\fwi_visionfm\data"),
            Path(r"D:\ryjin\fwi_visionfm\datasets"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _family_root(path: Path) -> Path:
    if path.parent.name.lower() in {"data", "model"}:
        return path.parent.parent
    return path.parent


def discover_families(data_root: str | Path) -> dict[str, list[dict[str, Any]]]:
    root = Path(data_root)
    data_files: dict[tuple[Path, int], Path] = {}
    model_files: dict[tuple[Path, int], Path] = {}
    for path in root.rglob("*.npy"):
        data_match = _DATA_PATTERN.match(path.name)
        if data_match:
            data_files[(_family_root(path), int(data_match.group("index")))] = path
        model_match = _MODEL_PATTERN.match(path.name)
        if model_match:
            model_files[(_family_root(path), int(model_match.group("index")))] = path
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    global_index = 0
    for family_root, file_id in sorted(set(data_files) & set(model_files), key=lambda item: (item[0].name, item[1])):
        data_path = data_files[(family_root, file_id)]
        model_path = model_files[(family_root, file_id)]
        data_array = np.load(data_path, mmap_mode="r")
        model_array = np.load(model_path, mmap_mode="r")
        if data_array.shape[0] != model_array.shape[0]:
            raise ValueError(f"sample count mismatch for {data_path} and {model_path}")
        for local_index in range(int(data_array.shape[0])):
            families[family_root.name].append(
                {
                    "family": family_root.name,
                    "data_file": str(data_path),
                    "model_file": str(model_path),
                    "local_index": int(local_index),
                    "global_index": int(global_index),
                    "path": str(data_path),
                }
            )
            global_index += 1
    for sample_path in sorted(root.rglob("sample_*.npz")):
        family = sample_path.parent.name
        with np.load(sample_path) as sample:
            if "records" not in sample or "velocity" not in sample:
                continue
        families[family].append(
            {
                "family": family,
                "data_file": str(sample_path),
                "model_file": str(sample_path),
                "local_index": 0,
                "global_index": int(global_index),
                "path": str(sample_path),
            }
        )
        global_index += 1
    return {name: rows for name, rows in sorted(families.items())}


def select_fault_family(families: dict[str, list[dict[str, Any]]], required: int) -> str | None:
    preferred = ["FlatFault_A", "FlatFault_B", "CurveFault_A", "CurveFault_B", "Fault"]
    names = sorted(families)
    fault_like = [name for name in preferred if name in families] + [
        name for name in names if "fault" in name.lower() and name not in preferred
    ]
    for name in fault_like:
        if len(families[name]) >= required:
            return name
    return None


def _match_family(families: dict[str, list[dict[str, Any]]], canonical: str, required: int) -> str | None:
    if canonical in families and len(families[canonical]) >= required:
        return canonical
    needle = canonical.lower()
    candidates = [
        name for name in sorted(families)
        if (name.lower() == needle or name.lower().startswith(needle) or needle in name.lower()) and len(families[name]) >= required
    ]
    return candidates[0] if candidates else None


def _sample_rows(rows: list[dict[str, Any]], count: int, rng: np.random.Generator) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"family has {len(rows)} samples, need {count}")
    order = rng.permutation(len(rows))[:count]
    return [dict(rows[int(index)]) for index in order]


def _split_source(rows: list[dict[str, Any]], train_size: int, val_size: int, test_size: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    total = train_size + val_size + test_size
    selected = _sample_rows(rows, total, np.random.default_rng(seed))
    return selected[:train_size], selected[train_size : train_size + val_size], selected[train_size + val_size :]


def _write_split_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SPLIT_FIELDS)
        writer.writeheader()
        writer.writerows([{field: row[field] for field in _SPLIT_FIELDS} for row in rows])


def _compute_train_stats(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    input_sum = 0.0
    input_sq_sum = 0.0
    input_count = 0
    target_min = float("inf")
    target_max = float("-inf")
    source_files = set()
    for row in rows:
        data, velocity = _load_sample_arrays(row)
        data = data.astype(np.float64, copy=False)
        velocity = velocity.astype(np.float64, copy=False)
        input_sum += float(data.sum())
        input_sq_sum += float(np.square(data).sum())
        input_count += int(data.size)
        target_min = min(target_min, float(velocity.min()))
        target_max = max(target_max, float(velocity.max()))
        source_files.add(row["data_file"])
    mean = input_sum / max(input_count, 1)
    std = float(np.sqrt(max(input_sq_sum / max(input_count, 1) - mean * mean, 0.0)))
    payload = {
        "seismic": {"mean": float(mean), "std": std},
        "velocity": {"min": float(target_min), "max": float(target_max)},
        "input_mean": float(mean),
        "input_std": std,
        "target_min": float(target_min),
        "target_max": float(target_max),
        "sample_count": len(rows),
        "source_files": sorted(source_files),
        "stats_split": "source_train_only",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def _shape_payload(row: dict[str, Any]) -> tuple[list[int], list[int]]:
    data, velocity = _load_sample_arrays(row)
    if velocity.ndim == 2:
        velocity_shape = [1, int(velocity.shape[0]), int(velocity.shape[1])]
    else:
        velocity_shape = [int(v) for v in velocity.shape]
    return [int(v) for v in data.shape], velocity_shape


def _load_sample_arrays(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    data_path = Path(row["data_file"])
    if data_path.suffix.lower() == ".npz":
        with np.load(data_path) as payload:
            return np.asarray(payload["records"]), np.asarray(payload["velocity"])
    local_index = int(row["local_index"])
    return np.load(row["data_file"], mmap_mode="r")[local_index], np.load(row["model_file"], mmap_mode="r")[local_index]


def build_protocol_v2_splits(
    *,
    data_root: str | Path,
    output_root: str | Path,
    train_size: int = 500,
    val_size: int = 100,
    test_size: int = 100,
    seeds: list[int] | tuple[int, ...] = (0, 1, 2),
) -> dict[str, Any]:
    root = Path(data_root)
    output = Path(output_root)
    manifest_root = output / "manifests"
    families = discover_families(root)
    required_source = train_size + val_size + test_size
    required_target = test_size
    fault_family = select_fault_family(families, required_target)
    flatvel = _match_family(families, "FlatVel_A", required_source)
    curvevel_source = _match_family(families, "CurveVel_A", required_source)
    curvevel_target = _match_family(families, "CurveVel_A", required_target)
    requested_pairs = []
    if flatvel and curvevel_target:
        requested_pairs.append((flatvel, curvevel_target))
    if fault_family is not None:
        if flatvel:
            requested_pairs.append((flatvel, fault_family))
        if curvevel_source:
            requested_pairs.append((curvevel_source, fault_family))
    available_pairs = [
        (source, target)
        for source, target in requested_pairs
        if source in families and target in families and len(families[source]) >= required_source and len(families[target]) >= required_target
    ]
    if not available_pairs:
        raise ValueError(
            "no protocol v2 family pairs have enough samples; "
            f"families={ {name: len(rows) for name, rows in families.items()} }"
        )

    manifests = []
    for source, target in available_pairs:
        for seed in seeds:
            train, val, in_test = _split_source(families[source], train_size, val_size, test_size, int(seed))
            cross = _sample_rows(families[target], test_size, np.random.default_rng(int(seed) + 10_000))
            stem = f"{source}_to_{target}_seed{seed}"
            split_dir = manifest_root / stem
            paths = {
                "train": split_dir / "train.csv",
                "val": split_dir / "val.csv",
                "in_family_test": split_dir / "in_family_test.csv",
                "cross_family_test": split_dir / "cross_family_test.csv",
            }
            _write_split_csv(paths["train"], train)
            _write_split_csv(paths["val"], val)
            _write_split_csv(paths["in_family_test"], in_test)
            _write_split_csv(paths["cross_family_test"], cross)
            stats_path = split_dir / "train_stats.json"
            _compute_train_stats(train, stats_path)
            data_shape, velocity_shape = _shape_payload(train[0])
            manifest = {
                "protocol": "protocol_v2_small_benchmark",
                "source_family": source,
                "target_family": target,
                "seed": int(seed),
                "data_root": str(root),
                "data_shape": data_shape,
                "velocity_shape": velocity_shape,
                "stats_path": str(stats_path),
                "split_files": {name: str(path) for name, path in paths.items()},
                "train_samples": train,
                "val_samples": val,
                "in_family_test_samples": in_test,
                "cross_family_test_samples": cross,
                "normalization": "train-only stats from source train split",
            }
            manifest_path = manifest_root / f"{stem}_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            manifests.append(str(manifest_path))
    summary = {
        "data_root": str(root),
        "output_root": str(output),
        "families": {name: len(rows) for name, rows in families.items()},
        "fault_family": fault_family,
        "pair_count": len(available_pairs),
        "manifest_count": len(manifests),
        "manifests": manifests,
    }
    manifest_root.mkdir(parents=True, exist_ok=True)
    (manifest_root / "protocol_v2_manifest_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build protocol v2 OpenFWI cross-family splits.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/protocol_v2_small"))
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    if data_root is None:
        print("OpenFWI data root not found. Tried --data-root, OPENFWI_ROOT, D:\\ryjin\\OpenFWI, data, datasets.")
        return
    summary = build_protocol_v2_splits(
        data_root=data_root,
        output_root=args.output_root,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        seeds=args.seeds,
    )
    print(f"Wrote {summary['manifest_count']} manifests under {Path(args.output_root) / 'manifests'}")


if __name__ == "__main__":
    main()
