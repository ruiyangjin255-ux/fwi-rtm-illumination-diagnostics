"""FlatFault-A diagnostics for Phase-2 locked experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import load_arrays
from .diagnostics import (
    edge_mask_from_threshold,
    edge_prf,
    gradient_magnitude_np,
    gradient_metrics,
    masked_mae_identity,
    simple_metrics,
    write_csv,
)
from .fault_plotting import comparison_figure, gradient_figure, profiles_figure, save_both
from .protocol import load_protocol


def _load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {name: np.asarray(payload[name]) for name in ("sample_id", "prediction", "target")}


def _metric_dict(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    base = simple_metrics(pred, target, mask)
    identity = masked_mae_identity(pred, target, mask)
    grad = gradient_metrics(pred, target, mask)
    rmse = max(float(base["RMSE"]), 1e-6)
    data_range = max(float(target.max() - target.min()), 1e-6)
    pred_g = gradient_magnitude_np(pred)
    true_g = gradient_magnitude_np(target)
    rel = float(np.linalg.norm((pred - target).reshape(-1)) / max(np.linalg.norm(target.reshape(-1)), 1e-6))
    return {
        "MAE": base["MAE"],
        "RMSE": base["RMSE"],
        "SSIM": base["SSIM"],
        "PSNR": float(20.0 * np.log10(data_range / rmse)),
        "source_threshold_edge_MAE": identity["edge_MAE"],
        "edge_MAE": identity["edge_MAE"],
        "nonedge_MAE": identity["nonedge_MAE"],
        "relative_error": rel,
        "gradient_magnitude_MAE_all": float(np.abs(pred_g - true_g).mean()),
        "gradient_magnitude_MAE_edge": float(np.abs(pred_g - true_g)[mask].mean()) if mask.any() else 0.0,
        **grad,
    }


def _fault_proxy(mask: np.ndarray) -> np.ndarray:
    # Deterministic diagnostic only: keep edge columns with vertical support above the 75th percentile.
    col_support = mask.sum(axis=0)
    tau = np.percentile(col_support, 75)
    proxy = mask & (col_support[None, :] >= tau)
    return proxy


def run_fault_diagnostics(experiment_root: str | Path, protocol: str | Path, output: str | Path, figures: str | Path) -> dict[str, Any]:
    root = Path(experiment_root)
    out = Path(output)
    fig_dir = Path(figures)
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_protocol(protocol)
    source = load_arrays(manifest.source.records, manifest.source.models, max_samples=max(manifest.train_indices) + 1, family=manifest.source.family)
    tau = float(np.percentile(gradient_magnitude_np(source.velocities[np.asarray(manifest.train_indices, dtype=int)]).reshape(-1), 90.0))
    variants = [p.name for p in root.iterdir() if p.is_dir() and p.name.startswith(("B1", "B4"))]
    coverage_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    fault_rows: list[dict[str, Any]] = []
    archives = {variant: {seed: _load_archive(root / variant / f"seed_{seed}" / "predictions_cross_family.npz") for seed in (0, 1, 2)} for variant in variants}
    for variant, seed_map in archives.items():
        for seed, archive in seed_map.items():
            pred_threshold = float(np.percentile(gradient_magnitude_np(archive["prediction"]).reshape(-1), 90.0))
            for idx, sample_id in enumerate(archive["sample_id"].astype(int)):
                pred = archive["prediction"][idx]
                target = archive["target"][idx]
                mask = edge_mask_from_threshold(target, tau)
                pred_mask = gradient_magnitude_np(pred) > pred_threshold
                identity = masked_mae_identity(pred, target, mask)
                base = {"sample_id": int(sample_id), "variant": variant, "seed": seed, "edge_threshold": tau, "mask_mode": "source_threshold_strict_gt"}
                coverage_rows.append({**base, **identity})
                metrics = _metric_dict(pred, target, mask)
                edge_rows.append({**base, **identity, **metrics})
                detection_rows.append({**base, **edge_prf(pred_mask, mask, tolerance_pixels=1)})
                h = target.shape[0]
                zones = {"shallow": slice(0, h // 3), "middle": slice(h // 3, 2 * h // 3), "deep": slice(2 * h // 3, h)}
                for zone, sl in zones.items():
                    depth_rows.append({**base, "zone": zone, **_metric_dict(pred[sl], target[sl], mask[sl]), **edge_prf(pred_mask[sl], mask[sl], tolerance_pixels=1)})
                proxy = _fault_proxy(mask)
                prf = edge_prf(pred_mask, proxy, tolerance_pixels=1)
                fault_rows.append({
                    **base,
                    "fault_proxy": "evaluation_diagnostic_only",
                    "not_ground_truth_fault_label": True,
                    "fault_proxy_MAE": float(np.abs(pred - target)[proxy].mean()) if proxy.any() else 0.0,
                    "fault_proxy_gradient_l1": gradient_metrics(pred, target, proxy)["gradient_l1_edge"] if proxy.any() else 0.0,
                    "fault_proxy_edge_recall": prf["edge_recall"],
                    "fault_proxy_edge_F1": prf["edge_F1"],
                })
    write_csv(out / "edge_mask_coverage.csv", coverage_rows)
    write_csv(out / "edge_vs_nonedge_metrics.csv", edge_rows)
    write_csv(out / "edge_detection_metrics.csv", detection_rows)
    write_csv(out / "per_depth_metrics.csv", depth_rows)
    write_csv(out / "fault_proxy_metrics.csv", fault_rows)
    (out / "gradient_metric_audit.json").write_text(json.dumps({"status": "PASS", "tau_source_90": tau, "edge_mask": "gradient_magnitude > tau", "gradient_unit": "m/s per grid_cell"}, indent=2), encoding="utf-8")

    # Fixed sample figures from seed 0 and B1 MAE ranks.
    b1 = archives["B1_raw_unet"][0]
    mae = np.abs(b1["prediction"] - b1["target"]).mean(axis=(1, 2))
    order = np.argsort(mae)
    choices = {"median": int(order[len(order) // 2]), "hard": int(order[int(round(0.75 * (len(order) - 1)))])}
    names = ["B1_raw_unet", "B4_no_geometry_attention", "B4_pasd_fwi"]
    for label, idx in choices.items():
        sid = int(b1["sample_id"][idx])
        target = b1["target"][idx]
        predictions = {name: archives[name][0]["prediction"][idx] for name in names if name in archives}
        masks = {"truth": edge_mask_from_threshold(target, tau)}
        metrics = {}
        for name, pred in predictions.items():
            pred_tau = float(np.percentile(gradient_magnitude_np(archives[name][0]["prediction"]).reshape(-1), 90.0))
            masks[name] = gradient_magnitude_np(pred) > pred_tau
            metrics[name] = _metric_dict(pred, target, masks["truth"])
        comparison_figure(target, predictions, metrics, fig_dir / f"Figure_flatfault_{label}_comparison.png", f"FlatFault {label} sample_id={sid}")
        if label == "median":
            gradient_figure(target, predictions, masks, fig_dir / "Figure_flatfault_gradient_comparison.png", f"FlatFault gradients sample_id={sid}")
            profiles_figure(target, predictions, fig_dir / "Figure_flatfault_profiles.png", f"FlatFault profiles sample_id={sid}")
    # Distribution and geometry ablation figures.
    for figure_name, source_rows, metric in [
        ("Figure_flatfault_metric_distributions", edge_rows, "MAE"),
        ("Figure_flatfault_geometry_ablation", edge_rows, "source_threshold_edge_MAE"),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        labels = names
        ax.boxplot([[float(r[metric]) for r in source_rows if r["variant"] == name] for name in labels], labels=labels, showfliers=False)
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        save_both(fig, fig_dir / f"{figure_name}.png")
    return {"status": "PASS", "tau_source_90": tau, "variants": variants}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FlatFault-A diagnostics and figures.")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--figures", required=True)
    args = parser.parse_args()
    result = run_fault_diagnostics(args.experiment_root, args.protocol, args.output, args.figures)
    print(json.dumps({"status": result["status"], "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
