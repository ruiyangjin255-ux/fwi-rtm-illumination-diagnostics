"""Create publication-ready common-sample figures across PASD variants without cherry-picking."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .metrics import per_sample_metrics
from .plotting import plot_gradient_comparison, plot_profiles, plot_velocity_comparison


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        required = {"sample_id", "prediction", "target"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Archive {path} missing {sorted(missing)}")
        return {name: np.asarray(archive[name]) for name in required}


def _archives_from_root(root: Path, variants: list[str], seed: int, split: str) -> dict[str, dict[str, np.ndarray]]:
    archives: dict[str, dict[str, np.ndarray]] = {}
    for variant in variants:
        path = root / variant / f"seed_{seed}" / f"predictions_{split}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Expected archive not found: {path}")
        archives[variant] = _load(path)
    return archives


def _aligned(archives: Mapping[str, dict[str, np.ndarray]]) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    first_name = next(iter(archives))
    first = archives[first_name]
    ids = np.asarray(first["sample_id"], dtype=np.int64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"Duplicate sample IDs in {first_name} archive.")
    expected = set(ids.tolist())
    predictions: dict[str, np.ndarray] = {}
    target: np.ndarray | None = None
    ordered_ids = np.sort(ids)
    for variant, archive in archives.items():
        mapping = {int(sample_id): index for index, sample_id in enumerate(archive["sample_id"].tolist())}
        if set(mapping) != expected:
            raise ValueError(f"Archive sample_id mismatch for {variant}; common sample selection requires exact alignment.")
        indices = np.asarray([mapping[int(sample_id)] for sample_id in ordered_ids])
        current_target = np.asarray(archive["target"])[indices]
        if target is None:
            target = current_target
        elif not np.allclose(target, current_target, rtol=0.0, atol=1e-5):
            raise ValueError(f"Targets differ across variants for split alignment: {variant}.")
        predictions[variant] = np.asarray(archive["prediction"])[indices]
    assert target is not None
    return ordered_ids, predictions, target


def _select_index(reference_prediction: np.ndarray, target: np.ndarray, selection: str, index: int | None) -> int:
    if index is not None:
        if index < 0 or index >= len(target):
            raise IndexError("--index is outside the selected split range.")
        return index
    metrics = per_sample_metrics(torch.from_numpy(reference_prediction), torch.from_numpy(target))["mae"].numpy()
    ordering = np.argsort(metrics)
    if selection == "best_mae":
        return int(ordering[0])
    if selection == "worst_mae":
        return int(ordering[-1])
    if selection == "median_mae":
        return int(ordering[len(ordering) // 2])
    raise ValueError(f"Unsupported selection policy: {selection}")


def _save_metric_rows(path: Path, sample_id: int, predictions: Mapping[str, np.ndarray], target: np.ndarray) -> None:
    rows: list[dict[str, object]] = []
    target_batch = torch.from_numpy(target[None])
    for variant, prediction in predictions.items():
        metrics = per_sample_metrics(torch.from_numpy(prediction[None]), target_batch)
        rows.append({"sample_id": sample_id, "variant": variant, **{name: float(value[0]) for name, value in metrics.items()}})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a fixed aligned test sample across PASD B1--B4 variants.")
    parser.add_argument("--protocol-root", required=True, help="Directory produced by run_protocol.py.")
    parser.add_argument("--variants", nargs="+", default=["B1_raw_unet", "B2_hybrid_unet", "B3_raw_bed", "B4_pasd_fwi"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["in_family", "cross_family"], default="cross_family")
    parser.add_argument("--selection", choices=["best_mae", "median_mae", "worst_mae"], default="median_mae")
    parser.add_argument("--index", type=int, default=None, help="Explicit index after stable sample_id sorting; bypasses selection policy.")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.protocol_root)
    output = Path(args.output) if args.output else root / "comparison" / f"seed_{args.seed}_{args.split}_{args.selection}"
    output.mkdir(parents=True, exist_ok=True)
    archives = _archives_from_root(root, args.variants, args.seed, args.split)
    ids, predictions, target = _aligned(archives)
    reference = predictions[args.variants[0]]
    choice = _select_index(reference, target, args.selection, args.index)
    sample_id = int(ids[choice])
    chosen_predictions = {variant: prediction[choice] for variant, prediction in predictions.items()}
    chosen_target = target[choice]
    plot_velocity_comparison(chosen_target, chosen_predictions, output / "velocity_comparison.png", title=f"{args.split}, sample_id={sample_id}")
    plot_profiles(chosen_target, chosen_predictions, output / "velocity_profiles.png")
    plot_gradient_comparison(chosen_target, chosen_predictions, output / "gradient_comparison.png")
    _save_metric_rows(output / "sample_metrics.csv", sample_id, chosen_predictions, chosen_target)
    metadata = {
        "split": args.split,
        "seed": args.seed,
        "selection": args.selection if args.index is None else "explicit_index",
        "selected_sorted_index": choice,
        "sample_id": sample_id,
        "variants": args.variants,
        "selection_reference": args.variants[0],
    }
    (output / "selection.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "SUCCESS", "output": str(output), **metadata}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
