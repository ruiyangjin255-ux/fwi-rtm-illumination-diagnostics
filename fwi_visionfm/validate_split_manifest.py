from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from fwi_visionfm.shape_utils import infer_split_manifest_shape
from fwi_visionfm.split_utils import load_split_paths, read_json, write_json


def validate_split_manifest(split_manifest: str | Path) -> dict[str, Any]:
    manifest_path = Path(split_manifest)
    payload = read_json(manifest_path)
    split_paths = load_split_paths(manifest_path)
    all_paths = [Path(path).resolve() for paths in split_paths.values() for path in paths]
    duplicates = [str(path) for path, count in Counter(all_paths).items() if count > 1]
    missing_paths = [str(path) for path in all_paths if not path.exists()]
    if missing_paths:
        raise FileNotFoundError(f"split manifest 包含不存在的样本路径: {missing_paths[:5]}")
    shape_summary = infer_split_manifest_shape(manifest_path)
    if not shape_summary["is_uniform_records_shape"]:
        raise ValueError(f"records 形状不一致: {shape_summary['records_shape_set']}")
    if not shape_summary["is_uniform_velocity_shape"]:
        raise ValueError(f"velocity 形状不一致: {shape_summary['velocity_shape_set']}")

    family_distribution: dict[str, dict[str, int]] = {}
    family_meta = payload.get("families", {})
    for family_name, meta in family_meta.items():
        family_distribution.setdefault(family_name, {"train": 0, "val": 0, "test": 0, "total": 0})
    for split_name, paths in split_paths.items():
        for sample_path in paths:
            family_name = "unknown"
            for candidate_family, meta in family_meta.items():
                dataset_names = meta.get("datasets", [])
                subset_names = meta.get("subsets", [])
                raw_text = str(sample_path).lower()
                if any(name.lower() in raw_text for name in dataset_names + subset_names + [candidate_family]):
                    family_name = candidate_family
                    break
            bucket = family_distribution.setdefault(family_name, {"train": 0, "val": 0, "test": 0, "total": 0})
            bucket[split_name] += 1
            bucket["total"] += 1

    summary = {
        "split_manifest": str(manifest_path),
        "counts": {name: len(paths) for name, paths in split_paths.items()},
        "train_count": len(split_paths["train"]),
        "val_count": len(split_paths["val"]),
        "test_count": len(split_paths["test"]),
        "duplicate_count": len(duplicates),
        "duplicates": duplicates,
        "family_distribution": family_distribution,
        "records_shape_set": shape_summary["records_shape_set"],
        "velocity_shape_set": shape_summary["velocity_shape_set"],
        "inferred_depth": shape_summary["inferred_depth"],
        "inferred_width": shape_summary["inferred_width"],
    }
    output_path = manifest_path.parent / "split_validation_summary.json"
    write_json(output_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="验证 split manifest 的路径、重复和形状一致性。")
    parser.add_argument("--split-manifest", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_split_manifest(args.split_manifest)
    print(f"写出 split 验证摘要: {args.split_manifest.parent / 'split_validation_summary.json'}")
    print(f"train/val/test: {summary['counts']['train']}/{summary['counts']['val']}/{summary['counts']['test']}")
    print(f"inferred depth/width: {summary['inferred_depth']}/{summary['inferred_width']}")


if __name__ == "__main__":
    main()
