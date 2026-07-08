"""Create publication-ready common-sample figures across PASD variants without cherry-picking."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .diagnostics import read_csv, write_csv
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
    parser.add_argument("--protocol-root", default=None, help="Directory produced by run_protocol.py.")
    parser.add_argument("--phase1-root", default=None)
    parser.add_argument("--experiment-root", default=None)
    parser.add_argument("--audit-root", default=None)
    parser.add_argument("--variants", nargs="+", default=["B1_raw_unet", "B2_hybrid_unet", "B3_raw_bed", "B4_pasd_fwi"])
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["in_family", "cross_family"], default="cross_family")
    parser.add_argument("--selection", choices=["best_mae", "median_mae", "worst_mae"], default="median_mae")
    parser.add_argument("--index", type=int, default=None, help="Explicit index after stable sample_id sorting; bypasses selection policy.")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def _metric_maps(audit_root: Path) -> dict[tuple[str, int, int], dict[str, float]]:
    rows = read_csv(audit_root / "edge_vs_nonedge_metrics.csv")
    out: dict[tuple[str, int, int], dict[str, float]] = {}
    for row in rows:
        if row.get("mask_mode") != "source_threshold" or str(row.get("edge_percentile")) != "90":
            continue
        key = (row["variant"], int(row["seed"]), int(row["sample_id"]))
        out[key] = {
            "MAE": float(row["full_MAE"]),
            "edge_MAE": float(row["edge_MAE"]),
            "gradient_l1_all": float(row["gradient_l1_all"]),
            "gradient_l1_edge": float(row["gradient_l1_edge"]),
            "edge_F1": float(row["edge_F1"]),
        }
    return out


def _archive_maps(root: Path, variants: list[str]) -> dict[tuple[str, int, int], dict[str, float]]:
    out: dict[tuple[str, int, int], dict[str, float]] = {}
    for variant in variants:
        for seed in (0, 1, 2):
            path = root / variant / f"seed_{seed}" / "predictions_cross_family.npz"
            with np.load(path) as payload:
                ids = payload["sample_id"].astype(int)
                metrics = per_sample_metrics(torch.from_numpy(payload["prediction"]), torch.from_numpy(payload["target"]))
                for idx, sample_id in enumerate(ids.tolist()):
                    out[(variant, seed, sample_id)] = {
                        "RMSE": float(metrics["rmse"][idx]),
                        "SSIM": float(metrics["ssim"][idx]),
                    }
    return out


def _bootstrap_delta(delta: np.ndarray, metric: str, n: int, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(0, len(delta), size=(int(n), len(delta)))].mean(axis=1)
    lower_is_better = metric != "SSIM" and metric != "edge_F1"
    improvement = -delta if lower_is_better else delta
    return {
        "mean": float(delta.mean()),
        "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "improvement_probability": float((improvement > 0).mean()),
        "n": int(len(delta)),
    }


def run_component_attribution(phase1_root: str | Path, audit_root: str | Path, output: str | Path, variants: list[str], bootstrap_resamples: int) -> None:
    phase1_root = Path(phase1_root)
    audit_root = Path(audit_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics = _metric_maps(audit_root)
    metrics2 = _archive_maps(phase1_root, variants)
    for key, values in metrics2.items():
        metrics.setdefault(key, {}).update(values)
    comparisons = {
        "B2_minus_B1": ("B2_hybrid_unet", "B1_raw_unet", "Hybrid bridge contribution"),
        "B3_minus_B1": ("B3_raw_bed", "B1_raw_unet", "Background-edge decoder/loss contribution"),
        "B4_minus_B2": ("B4_pasd_fwi", "B2_hybrid_unet", "Decoder/loss/geometry attention extra contribution"),
        "B4_minus_B3": ("B4_pasd_fwi", "B3_raw_bed", "Hybrid bridge plus geometry extra contribution"),
        "B4_minus_B1": ("B4_pasd_fwi", "B1_raw_unet", "Full PASD-FWI total contribution"),
    }
    metric_names = ["MAE", "RMSE", "SSIM", "edge_MAE", "gradient_l1_all", "gradient_l1_edge", "edge_F1"]
    rows: list[dict[str, object]] = []
    boot: dict[str, object] = {}
    for comparison_id, (candidate, baseline, meaning) in comparisons.items():
        for seed_label in [0, 1, 2, "combined"]:
            sample_ids = sorted({
                key[2] for key in metrics
                if key[0] == candidate and (seed_label == "combined" or key[1] == seed_label)
            }.intersection({
                key[2] for key in metrics
                if key[0] == baseline and (seed_label == "combined" or key[1] == seed_label)
            }))
            for metric in metric_names:
                deltas = []
                for key in sorted(metrics):
                    variant, seed, sample_id = key
                    if variant != candidate or sample_id not in sample_ids or (seed_label != "combined" and seed != seed_label):
                        continue
                    base_key = (baseline, seed, sample_id)
                    if base_key in metrics and metric in metrics[key] and metric in metrics[base_key]:
                        deltas.append(float(metrics[key][metric]) - float(metrics[base_key][metric]))
                if not deltas:
                    continue
                delta = np.asarray(deltas, dtype=np.float64)
                stat = _bootstrap_delta(delta, metric, bootstrap_resamples, seed=sum(map(ord, comparison_id + metric + str(seed_label))))
                row = {
                    "comparison": comparison_id,
                    "meaning": meaning,
                    "seed": seed_label,
                    "metric": metric,
                    "delta": stat["mean"],
                    "ci_low": stat["ci95"][0],
                    "ci_high": stat["ci95"][1],
                    "improvement_probability": stat["improvement_probability"],
                    "n_pairs": stat["n"],
                }
                rows.append(row)
                boot[f"{comparison_id}_{seed_label}_{metric}"] = stat
    write_csv(output / "component_attribution.csv", rows)
    (output / "component_attribution_bootstrap.json").write_text(json.dumps(boot, indent=2, ensure_ascii=False), encoding="utf-8")
    selected = [row for row in rows if row["seed"] == "combined" and row["metric"] in {"MAE", "SSIM", "edge_MAE", "gradient_l1_edge", "edge_F1"}]
    labels = sorted({str(row["comparison"]) for row in selected})
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), constrained_layout=True)
    for ax, metric in zip(axes, ["MAE", "SSIM", "edge_MAE", "gradient_l1_edge", "edge_F1"]):
        vals = [float(next(row["delta"] for row in selected if row["comparison"] == label and row["metric"] == metric)) for label in labels]
        ax.bar(np.arange(len(labels)), vals)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=8)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(output / "Figure_component_attribution.png", dpi=220)
    fig.savefig(output / "Figure_component_attribution.pdf")
    plt.close(fig)


def run_experiment_bootstrap(experiment_root: str | Path, variants: list[str], output: str | Path, bootstrap_resamples: int) -> None:
    root = Path(experiment_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    diag = root.parent / "diagnostics" / "edge_vs_nonedge_metrics.csv"
    edge_rows = read_csv(diag) if diag.exists() else []
    edge_map: dict[tuple[str, int, int], dict[str, float]] = {}
    for row in edge_rows:
        edge_map[(row["variant"], int(row["seed"]), int(row["sample_id"]))] = {
            "source_threshold_edge_MAE": float(row.get("source_threshold_edge_MAE", row.get("edge_MAE", 0.0))),
            "gradient_l1_edge": float(row["gradient_l1_edge"]),
        }
    det_path = root.parent / "diagnostics" / "edge_detection_metrics.csv"
    for row in read_csv(det_path) if det_path.exists() else []:
        edge_map.setdefault((row["variant"], int(row["seed"]), int(row["sample_id"])), {})["edge_F1"] = float(row["edge_F1"])
    metric_map: dict[tuple[str, int, int], dict[str, float]] = {}
    for variant in variants:
        for seed in (0, 1, 2):
            with np.load(root / variant / f"seed_{seed}" / "predictions_cross_family.npz") as payload:
                ids = payload["sample_id"].astype(int)
                metrics = per_sample_metrics(torch.from_numpy(payload["prediction"]), torch.from_numpy(payload["target"]))
                for i, sid in enumerate(ids.tolist()):
                    key = (variant, seed, sid)
                    metric_map[key] = {
                        "MAE": float(metrics["mae"][i]),
                        "RMSE": float(metrics["rmse"][i]),
                        "SSIM": float(metrics["ssim"][i]),
                    }
                    metric_map[key].update(edge_map.get(key, {}))
    comparisons = {
        "B4_pasd_fwi_vs_B1_raw_unet": ("B4_pasd_fwi", "B1_raw_unet"),
        "B4_pasd_fwi_vs_B4_no_geometry_attention": ("B4_pasd_fwi", "B4_no_geometry_attention"),
        "B4_no_geometry_attention_vs_B1_raw_unet": ("B4_no_geometry_attention", "B1_raw_unet"),
    }
    metrics = ["MAE", "RMSE", "SSIM", "source_threshold_edge_MAE", "gradient_l1_edge", "edge_F1"]
    summary: dict[str, object] = {}
    for comp, (candidate, baseline) in comparisons.items():
        comp_payload: dict[str, object] = {"candidate": candidate, "baseline": baseline, "seed_specific": {}, "pooled": {}}
        for seed_label in [0, 1, 2, "pooled"]:
            seed_payload: dict[str, object] = {}
            for metric in metrics:
                deltas = []
                for key, cvals in metric_map.items():
                    variant, seed, sid = key
                    if variant != candidate or (seed_label != "pooled" and seed != seed_label):
                        continue
                    bkey = (baseline, seed, sid)
                    if bkey in metric_map and metric in cvals and metric in metric_map[bkey]:
                        deltas.append(float(cvals[metric]) - float(metric_map[bkey][metric]))
                if not deltas:
                    continue
                stat = _bootstrap_delta(np.asarray(deltas), metric, bootstrap_resamples, seed=sum(map(ord, comp + metric + str(seed_label))))
                seed_payload[f"delta_{metric}"] = stat
            if seed_label == "pooled":
                comp_payload["pooled"] = seed_payload
            else:
                comp_payload["seed_specific"][str(seed_label)] = seed_payload
        (output / f"{comp}.json").write_text(json.dumps(comp_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        summary[comp] = comp_payload
    (output / "bootstrap_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.experiment_root:
        run_experiment_bootstrap(args.experiment_root, args.variants, args.output, args.bootstrap_resamples)
        print(json.dumps({"status": "SUCCESS", "output": args.output}, ensure_ascii=False, indent=2))
        return
    if args.phase1_root:
        if not args.audit_root:
            raise SystemExit("--audit-root is required with --phase1-root")
        run_component_attribution(args.phase1_root, args.audit_root, args.output, args.variants, args.bootstrap_resamples)
        print(json.dumps({"status": "SUCCESS", "output": args.output}, ensure_ascii=False, indent=2))
        return
    if not args.protocol_root:
        raise SystemExit("--protocol-root is required unless --phase1-root is used")
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
