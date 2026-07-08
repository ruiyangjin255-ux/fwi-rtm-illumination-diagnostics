from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fwi_visionfm.split_utils import collect_dataset_samples, materialize_split_payload, split_records, write_json


def make_cross_family_split(
    *,
    train_dirs: list[str | Path],
    test_dirs: list[str | Path],
    output_path: str | Path,
    train_fraction: float = 0.8,
    val_fraction: float = 0.0,
    seed: int = 2026,
) -> dict[str, Any]:
    train_records = collect_dataset_samples(train_dirs)
    test_records = collect_dataset_samples(test_dirs)
    split_map = split_records(train_records, train_fraction=train_fraction, val_fraction=val_fraction, seed=seed)
    split_map["test"] = test_records
    payload = materialize_split_payload(
        split_map,
        seed=seed,
        mode="cross_family",
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    payload["train_dirs"] = [str(Path(path)) for path in train_dirs]
    payload["test_dirs"] = [str(Path(path)) for path in test_dirs]
    write_json(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 cross-family split manifest。train-dirs 内部再划分 train/val，test-dirs 全部进入 test。")
    parser.add_argument("--train-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--test-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = make_cross_family_split(
        train_dirs=args.train_dirs,
        test_dirs=args.test_dirs,
        output_path=args.output,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(f"写出 cross-family split: {args.output}")
    print(f"train/val/test: {len(payload['train'])}/{len(payload['val'])}/{len(payload['test'])}")


if __name__ == "__main__":
    main()
