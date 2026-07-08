from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fwi_visionfm.split_utils import collect_dataset_samples, materialize_split_payload, split_records, write_json


def make_split_manifest(
    *,
    data_dirs: list[str | Path],
    output_path: str | Path,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 2026,
) -> dict[str, Any]:
    sample_records = collect_dataset_samples(data_dirs)
    split_map = split_records(sample_records, train_fraction=train_fraction, val_fraction=val_fraction, seed=seed)
    payload = materialize_split_payload(
        split_map,
        seed=seed,
        mode="in_domain",
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )
    payload["data_dirs"] = [str(Path(path)) for path in data_dirs]
    write_json(output_path, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据本地逐样本 npz 数据目录生成固定 split manifest。")
    parser.add_argument("--data-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = make_split_manifest(
        data_dirs=args.data_dirs,
        output_path=args.output,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    print(f"写出 split manifest: {args.output}")
    print(f"train/val/test: {len(payload['train'])}/{len(payload['val'])}/{len(payload['test'])}")


if __name__ == "__main__":
    main()
