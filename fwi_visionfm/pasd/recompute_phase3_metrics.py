"""Recompute Phase-3R official corrected metrics from fresh prediction archives only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

from .corrected_metrics import archive_arrays, archive_sha, corrected_sample_metrics, source_threshold_from_config, stable_id_hash
from .diagnostics import gradient_magnitude_np
from .phase3_utils import load_json, write_json


DATASETS = {"in_family": "in_family", "cross_curvevel_a": "curvevel_a", "cross_flatfault_a": "flatfault_a"}
VARIANTS = ("B1_raw_unet", "PASD_Core_locked")
SEEDS = (0, 1, 2)
METRICS = (
    "MAE",
    "RMSE",
    "Relative_Error",
    "SSIM",
    "PSNR",
    "source_threshold_edge_MAE",
    "nonedge_MAE",
    "gradient_l1_all",
    "gradient_l1_edge",
    "gradient_magnitude_MAE_all",
    "gradient_magnitude_MAE_edge",
    "edge_precision",
    "edge_recall",
    "edge_F1",
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _archive_path(root: Path, variant: str, seed: int, dataset: str) -> Path:
    return root / "dual_target_formal" / "prediction_archives" / f"{variant}_seed{seed}_{dataset}.npz"


def _prediction_edge_threshold(phase3_root: Path, selected_candidate: str) -> dict[str, Any]:
    grads: list[np.ndarray] = []
    source = phase3_root / "source_aggregation_selection" / "runs" / selected_candidate
    for archive in sorted(source.glob("seed_*/predictions_val.npz")):
        arrays = archive_arrays(archive)
        grads.append(gradient_magnitude_np(arrays["prediction"]))
    if not grads:
        raise FileNotFoundError(f"No source validation archives found under {source}")
    value = float(np.percentile(np.concatenate([g.reshape(-1) for g in grads]), 90.0))
    return {
        "threshold_fitting_split": "source_val",
        "threshold_fitting_target_access": False,
        "threshold_value": value,
        "selection_objective": "90th percentile prediction gradient on locked source validation predictions for selected PASD-Core candidate",
        "source_archives": [str(path) for path in sorted(source.glob("seed_*/predictions_val.npz"))],
    }


def recompute_metrics(args: argparse.Namespace) -> Path:
    if not args.use_fresh_prediction_archives_only:
        raise ValueError("--use-fresh-prediction-archives-only is required.")
    if args.edge_mask != "source_threshold_strict_gt":
        raise ValueError("Phase-3R requires --edge-mask source_threshold_strict_gt.")
    if args.prediction_edge_threshold != "source_val_locked":
        raise ValueError("Phase-3R requires --prediction-edge-threshold source_val_locked.")
    phase3_root = Path(args.phase3_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = load_json(args.locked_config)
    tau_source = source_threshold_from_config(config)
    pred_threshold = _prediction_edge_threshold(phase3_root, str(config["selected_candidate"]))
    pred_threshold["config_hash"] = config.get("phase1b_locked_config_sha256", config.get("config_hash", ""))
    tau_pred = float(pred_threshold["threshold_value"])
    write_json(out / "source_edge_threshold.json", {"tau_source": tau_source, "mask_condition": "strict_gt", "source": "locked Phase-1b config"})
    write_json(out / "prediction_edge_threshold_source_val.json", pred_threshold)
    all_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    for dataset, subdir in DATASETS.items():
        dataset_rows: list[dict[str, Any]] = []
        for variant in VARIANTS:
            for seed in SEEDS:
                archive_path = _archive_path(phase3_root, variant, seed, dataset)
                arrays = archive_arrays(archive_path)
                sample_ids = arrays["sample_id"].astype(np.int64)
                for index, sample_id in enumerate(sample_ids.tolist()):
                    metrics, coverage = corrected_sample_metrics(
                        arrays["prediction"][index],
                        arrays["target"][index],
                        tau_source=tau_source,
                        tau_pred=tau_pred,
                        dx=1.0,
                        dz=1.0,
                    )
                    reconstructed = coverage["edge_coverage"] * metrics["source_threshold_edge_MAE"] + (1.0 - coverage["edge_coverage"]) * metrics["nonedge_MAE"]
                    identity_error = abs(metrics["MAE"] - reconstructed)
                    if identity_error > 1e-5:
                        status = "FAILED"
                    else:
                        status = "PASSED"
                    base = {
                        "dataset": dataset,
                        "target_family": subdir,
                        "variant": variant,
                        "seed": seed,
                        "sample_id": int(sample_id),
                        "metric_source": "fresh_prediction_archive",
                        "archive_path": str(archive_path),
                        "archive_sha256": archive_sha(archive_path),
                        "sample_id_hash": stable_id_hash(sample_ids),
                        "gradient_unit": "m/s per grid_cell",
                        "dx": 1.0,
                        "dz": 1.0,
                        "mask_condition": "strict_gt",
                    }
                    row = {**base, **metrics}
                    dataset_rows.append(row)
                    all_rows.append(row)
                    coverage_rows.append({**base, **coverage})
                    identity_rows.append(
                        {
                            **base,
                            "full_MAE": metrics["MAE"],
                            "edge_coverage": coverage["edge_coverage"],
                            "edge_MAE": metrics["source_threshold_edge_MAE"],
                            "nonedge_MAE": metrics["nonedge_MAE"],
                            "reconstructed_MAE": reconstructed,
                            "absolute_identity_error": identity_error,
                            "status": status,
                        }
                    )
                    if status != "PASSED":
                        raise ValueError(f"Masked-MAE identity failed for {archive_path} sample {sample_id}: {identity_error}")
        dataset_out = out / subdir
        _write_csv(dataset_out / "corrected_per_sample_metrics.csv", dataset_rows)
        _write_csv(dataset_out / "corrected_summary_by_seed.csv", _summary(dataset_rows, ("dataset", "variant", "seed")))
        _write_csv(dataset_out / "corrected_summary_across_seeds.csv", _summary(_summary(dataset_rows, ("dataset", "variant", "seed")), ("dataset", "variant")))
    _write_csv(out / "all_corrected_per_sample_metrics.csv", all_rows)
    _write_csv(out / "edge_mask_coverage.csv", coverage_rows)
    _write_csv(out / "masked_mae_identity_check.csv", identity_rows)
    _write_csv(out / "corrected_summary_by_seed.csv", _summary(all_rows, ("dataset", "variant", "seed")))
    _write_csv(out / "corrected_summary_across_seeds.csv", _summary(_summary(all_rows, ("dataset", "variant", "seed")), ("dataset", "variant")))
    _write_curvevel_diagnosis(out, phase3_root, all_rows, coverage_rows)
    return out


def _summary(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    out = []
    for group, members in sorted(groups.items(), key=lambda item: item[0]):
        result = {key: value for key, value in zip(keys, group)}
        for metric in METRICS:
            values = [float(row[metric]) for row in members if metric in row and row[metric] != ""]
            if values:
                result[metric] = mean(values)
                result[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        out.append(result)
    return out


def _write_curvevel_diagnosis(out: Path, phase3_root: Path, rows: list[dict[str, Any]], coverage_rows: list[dict[str, Any]]) -> None:
    curve = [row for row in rows if row["dataset"] == "cross_curvevel_a"]
    curve_cov = [row for row in coverage_rows if row["dataset"] == "cross_curvevel_a"]
    b1 = [row for row in curve if row["variant"] == "B1_raw_unet"]
    pasd = [row for row in curve if row["variant"] == "PASD_Core_locked"]
    old_summary = phase3_root / "dual_target_formal" / "protocol_summary.csv"
    avg_cov = mean(float(row["edge_coverage"]) for row in curve_cov)
    lines = [
        "# CurveVel-A Edge Metric Diagnosis",
        "",
        "1. Phase-3 report 中的 CurveVel-A `edge_MAE` 来源于旧 `protocol_summary.csv` 的 legacy archive/per-sample metric 字段。",
        "2. 该字段被 Phase-3R 标记为 deprecated，不进入主表、主图或正式结论。",
        "3. 旧字段未携带 strict `> tau_source` provenance，不能证明使用了 Phase-1b 后定义的 edge mask。",
        "4. `tau_source` 来自 FlatVel-A source train，数值接近 1e-6；Phase-3R 仍严格使用 `gradient_magnitude > tau_source`。",
        f"5. Corrected CurveVel-A edge mask coverage 均值为 {avg_cov:.6f}，不接近 1。",
        f"6. Corrected B1 full MAE / edge MAE: {mean(float(r['MAE']) for r in b1):.6g} / {mean(float(r['source_threshold_edge_MAE']) for r in b1):.6g}。",
        f"7. Corrected PASD full MAE / edge MAE: {mean(float(r['MAE']) for r in pasd):.6g} / {mean(float(r['source_threshold_edge_MAE']) for r in pasd):.6g}。",
        f"8. Corrected gradient_l1_edge B1 -> PASD: {mean(float(r['gradient_l1_edge']) for r in b1):.6g} -> {mean(float(r['gradient_l1_edge']) for r in pasd):.6g}。",
        "9. 旧值与新值差异主要来自 metric provenance 和 mask/gradient 定义重算；Phase-3R 不使用旧 CSV 中的 `edge_MAE`/`gradient_error`。",
        "",
        f"Historical summary audited: `{old_summary}`",
    ]
    target = out / "curvevel_a" / "CURVEVEL_EDGE_METRIC_DIAGNOSIS.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase3-root", required=True, type=Path)
    parser.add_argument("--locked-config", required=True, type=Path)
    parser.add_argument("--dual-target-protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--use-fresh-prediction-archives-only", action="store_true")
    parser.add_argument("--edge-mask", required=True)
    parser.add_argument("--prediction-edge-threshold", required=True)
    parser.add_argument("--dx", default="auto")
    parser.add_argument("--dz", default="auto")
    args = parser.parse_args()
    out = recompute_metrics(args)
    print(json.dumps({"status": "SUCCESS", "output": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
