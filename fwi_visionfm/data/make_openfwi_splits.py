from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_FIELDS = ("data_file", "model_file", "local_index", "global_index", "family")


def _read_manifest_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(REQUIRED_FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"manifest must contain columns: {', '.join(REQUIRED_FIELDS)}")
        return [dict(row) for row in reader]


def _pick_train_family(families: list[str], train_family: str) -> str:
    if train_family != "auto":
        if train_family not in families:
            raise ValueError(f"train_family {train_family} not found in manifest families: {families}")
        return train_family
    for family in families:
        if "flatvel" in family.lower():
            return family
    return families[0]


def _sample_rows(rows: list[dict[str, str]], size: int, rng: np.random.Generator) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if size <= 0 or not rows:
        return [], list(rows)
    order = rng.permutation(len(rows))
    take = min(size, len(rows))
    chosen_idx = set(int(index) for index in order[:take])
    chosen = [rows[index] for index in range(len(rows)) if index in chosen_idx]
    remaining = [rows[index] for index in range(len(rows)) if index not in chosen_idx]
    return chosen, remaining


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in REQUIRED_FIELDS})


def _family_distribution(rows: list[dict[str, str]]) -> dict[str, int]:
    return dict(Counter(row["family"] for row in rows))


def make_openfwi_splits(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    train_family: str = "auto",
    train_size: int = 500,
    val_size: int = 100,
    test_size: int = 100,
    cross_family_size: int = 100,
    smoke_train_size: int = 32,
    smoke_val_size: int = 16,
    seed: int = 2026,
) -> dict[str, Any]:
    rows = _read_manifest_csv(manifest_path)
    if not rows:
        raise ValueError("manifest is empty")
    families = sorted({row["family"] for row in rows})
    chosen_train_family = _pick_train_family(families, train_family)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    rng = np.random.default_rng(seed)
    warnings: list[str] = []

    train_family_rows = list(grouped[chosen_train_family])
    train_rows, remaining = _sample_rows(train_family_rows, train_size, rng)
    val_rows, remaining = _sample_rows(remaining, val_size, rng)
    test_in_family_rows, _ = _sample_rows(remaining, test_size, rng)

    other_families = [family for family in families if family != chosen_train_family]
    test_cross_family_rows: list[dict[str, str]] = []
    if not other_families:
        warnings.append("Only one family found; cross-family split is empty.")
    else:
        for family in other_families:
            chosen, _ = _sample_rows(list(grouped[family]), min(cross_family_size, 500 - len(test_cross_family_rows)), rng)
            test_cross_family_rows.extend(chosen)
            if len(test_cross_family_rows) >= 500:
                break

    seen = set()
    for split_name, split_rows in {
        "train": train_rows,
        "val": val_rows,
        "test_in_family": test_in_family_rows,
        "test_cross_family": test_cross_family_rows,
    }.items():
        for row in split_rows:
            global_index = int(row["global_index"])
            if global_index in seen:
                raise ValueError(f"global_index overlap detected at split {split_name}: {global_index}")
            seen.add(global_index)

    smoke_train_rows = train_rows[: min(smoke_train_size, len(train_rows))]
    smoke_val_rows = val_rows[: min(smoke_val_size, len(val_rows))]

    out_dir = Path(output_dir)
    _write_csv(out_dir / "train.csv", train_rows)
    _write_csv(out_dir / "val.csv", val_rows)
    _write_csv(out_dir / "test_in_family.csv", test_in_family_rows)
    _write_csv(out_dir / "test_cross_family.csv", test_cross_family_rows)
    _write_csv(out_dir / "smoke_train.csv", smoke_train_rows)
    _write_csv(out_dir / "smoke_val.csv", smoke_val_rows)

    summary = {
        "manifest": str(Path(manifest_path)),
        "train_family": chosen_train_family,
        "families": families,
        "counts": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test_in_family": len(test_in_family_rows),
            "test_cross_family": len(test_cross_family_rows),
            "smoke_train": len(smoke_train_rows),
            "smoke_val": len(smoke_val_rows),
        },
        "family_distribution": {
            "train": _family_distribution(train_rows),
            "val": _family_distribution(val_rows),
            "test_in_family": _family_distribution(test_in_family_rows),
            "test_cross_family": _family_distribution(test_cross_family_rows),
        },
        "warnings": warnings,
        "seed": int(seed),
    }
    (out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed sample-level OpenFWI small splits from openfwi_manifest.csv.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-family", default="auto")
    parser.add_argument("--train-size", type=int, default=500)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--cross-family-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = make_openfwi_splits(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        train_family=args.train_family,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        cross_family_size=args.cross_family_size,
        seed=args.seed,
    )
    print(f"写出 split: {args.output_dir}")
    print(json.dumps(summary["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
