from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.split_utils import collect_dataset_samples, materialize_split_payload, write_json


PROTOCOL_NAME = "protocol_v1_matched_target_test"


def _pick(records: list[dict[str, Any]], *, train_count: int, val_count: int, test_count: int, seed: int) -> dict[str, list[dict[str, Any]]]:
    required = train_count + val_count + test_count
    if len(records) < required:
        raise ValueError(f"样本数不足: need {required}, got {len(records)}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    shuffled = [records[int(index)] for index in order]
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count : train_count + val_count + test_count],
    }


def _write_protocol_split(
    output_path: Path,
    *,
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
    seed: int,
    split_name: str,
    target_family: str,
    source_families: list[str],
    train_count: int,
    val_count: int,
    test_count: int,
) -> dict[str, Any]:
    payload = materialize_split_payload(
        {"train": train, "val": val, "test": test},
        seed=seed,
        mode="matched_target_test",
    )
    payload.update(
        {
            "protocol": PROTOCOL_NAME,
            "split_name": split_name,
            "target_family": target_family,
            "source_families": source_families,
            "requested_train_count": int(train_count),
            "requested_val_count": int(val_count),
            "requested_test_count": int(test_count),
        }
    )
    write_json(output_path, payload)
    return payload


def make_target_test_protocol(
    *,
    flatvel_dir: str | Path,
    curvevel_dir: str | Path,
    flatfault_dir: str | Path,
    output_dir: str | Path,
    train_count: int = 350,
    val_count: int = 50,
    test_count: int = 100,
    seed: int = 2026,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    family_records = {
        "flatvel_a": collect_dataset_samples([flatvel_dir]),
        "curvevel_a": collect_dataset_samples([curvevel_dir]),
        "flatfault_a": collect_dataset_samples([flatfault_dir]),
    }
    selected = {
        family: _pick(records, train_count=train_count, val_count=val_count, test_count=test_count, seed=seed + index)
        for index, (family, records) in enumerate(family_records.items())
    }

    half_train_a = train_count // 2
    half_train_b = train_count - half_train_a
    half_val_a = val_count // 2
    half_val_b = val_count - half_val_a

    split_specs = {
        "protocol_v1_flatvel_indomain": {
            "train": selected["flatvel_a"]["train"],
            "val": selected["flatvel_a"]["val"],
            "test": selected["flatvel_a"]["test"],
            "target": "flatvel_a",
            "sources": ["flatvel_a"],
        },
        "protocol_v1_curvevel_indomain": {
            "train": selected["curvevel_a"]["train"],
            "val": selected["curvevel_a"]["val"],
            "test": selected["curvevel_a"]["test"],
            "target": "curvevel_a",
            "sources": ["curvevel_a"],
        },
        "protocol_v1_flatfault_indomain": {
            "train": selected["flatfault_a"]["train"],
            "val": selected["flatfault_a"]["val"],
            "test": selected["flatfault_a"]["test"],
            "target": "flatfault_a",
            "sources": ["flatfault_a"],
        },
        "protocol_v1_flatvel_to_curvevel": {
            "train": selected["flatvel_a"]["train"],
            "val": selected["flatvel_a"]["val"],
            "test": selected["curvevel_a"]["test"],
            "target": "curvevel_a",
            "sources": ["flatvel_a"],
        },
        "protocol_v1_curvevel_to_flatvel": {
            "train": selected["curvevel_a"]["train"],
            "val": selected["curvevel_a"]["val"],
            "test": selected["flatvel_a"]["test"],
            "target": "flatvel_a",
            "sources": ["curvevel_a"],
        },
        "protocol_v1_flat_curve_to_flatfault": {
            "train": selected["flatvel_a"]["train"][:half_train_a] + selected["curvevel_a"]["train"][:half_train_b],
            "val": selected["flatvel_a"]["val"][:half_val_a] + selected["curvevel_a"]["val"][:half_val_b],
            "test": selected["flatfault_a"]["test"],
            "target": "flatfault_a",
            "sources": ["flatvel_a", "curvevel_a"],
        },
    }

    manifests: dict[str, str] = {}
    for name, spec in split_specs.items():
        path = output / f"{name}.json"
        _write_protocol_split(
            path,
            train=spec["train"],
            val=spec["val"],
            test=spec["test"],
            seed=seed,
            split_name=name,
            target_family=spec["target"],
            source_families=spec["sources"],
            train_count=train_count,
            val_count=val_count,
            test_count=test_count,
        )
        manifests[name] = str(path)

    summary = {
        "protocol": PROTOCOL_NAME,
        "seed": int(seed),
        "train_count": int(train_count),
        "val_count": int(val_count),
        "test_count": int(test_count),
        "manifests": manifests,
        "families": {family: len(records) for family, records in family_records.items()},
    }
    write_json(output / "protocol_v1_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 Protocol v1 matched target-test split manifests。")
    parser.add_argument("--flatvel-dir", required=True, type=Path)
    parser.add_argument("--curvevel-dir", required=True, type=Path)
    parser.add_argument("--flatfault-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-count", type=int, default=350)
    parser.add_argument("--val-count", type=int, default=50)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = make_target_test_protocol(
        flatvel_dir=args.flatvel_dir,
        curvevel_dir=args.curvevel_dir,
        flatfault_dir=args.flatfault_dir,
        output_dir=args.output_dir,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        seed=args.seed,
    )
    print(f"写出 Protocol v1 split manifests: {args.output_dir}")
    print(json.dumps(summary["manifests"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
