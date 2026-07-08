"""Audit PASD Phase-1 edge masks, gradient metrics, and branch diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import OpenFWINpyDataset, VelocityScaler
from .diagnostic_plotting import barplot, boxplot
from .diagnostics import (
    edge_mask_from_threshold,
    edge_mask_percentile,
    edge_prf,
    gaussian_smooth_np,
    global_ssim_np,
    gradient_magnitude_np,
    gradient_metrics,
    masked_mae_identity,
    simple_metrics,
    write_csv,
)
from .experiment import TrainingConfig, build_model
from .protocol import load_protocol, load_protocol_bundles
from .registry import get_variant


def _load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {name: np.asarray(payload[name]) for name in ("sample_id", "prediction", "target")}


def _training_config(payload: dict[str, Any]) -> TrainingConfig:
    data = dict(payload.get("training", {}))
    valid = TrainingConfig.__dataclass_fields__.keys()
    return TrainingConfig(**{key: data[key] for key in data if key in valid})


def _predict_split(run_dir: Path, bundle, indices: tuple[int, ...], variant_name: str, summary: dict[str, Any]) -> dict[str, np.ndarray]:
    config = _training_config(summary)
    scaler_payload = summary["train_only_scaler"]
    scaler = VelocityScaler(float(scaler_payload["minimum"]), float(scaler_payload["maximum"]))
    dataset = OpenFWINpyDataset(
        bundle.records,
        bundle.velocities,
        indices,
        scaler=scaler,
        sample_ids=bundle.sample_ids,
        source_positions=bundle.source_positions,
        receiver_positions=bundle.receiver_positions,
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    model = build_model(get_variant(variant_name), tuple(int(x) for x in bundle.velocities.shape[-2:]), config)
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    ids: list[np.ndarray] = []
    pred: list[np.ndarray] = []
    target: list[np.ndarray] = []
    background: list[np.ndarray] = []
    residual: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["records"].float(), source_positions=batch.get("source_positions"), receiver_positions=batch.get("receiver_positions"))
            ids.append(batch["sample_id"].numpy())
            pred.append(scaler.denormalize(out.velocity[:, 0].numpy()))
            target.append(scaler.denormalize(batch["velocity"][:, 0].numpy()))
            background.append(scaler.denormalize(out.background[:, 0].numpy()))
            residual.append((out.residual[:, 0].numpy() * (scaler.maximum - scaler.minimum)).astype(np.float32))
    return {
        "sample_id": np.concatenate(ids),
        "prediction": np.concatenate(pred),
        "target": np.concatenate(target),
        "background": np.concatenate(background),
        "residual": np.concatenate(residual),
    }


def _source_thresholds(source_velocities: np.ndarray, train_indices: tuple[int, ...], percentiles: list[float], dx: float, dz: float) -> dict[str, float]:
    gradients = gradient_magnitude_np(source_velocities[np.asarray(train_indices, dtype=np.int64)], dx=dx, dz=dz).reshape(-1)
    return {f"tau_source_{int(p)}": float(np.percentile(gradients, p)) for p in percentiles}


def _depth_zone_slices(h: int) -> dict[str, slice]:
    return {"shallow": slice(0, h // 3), "middle": slice(h // 3, (2 * h) // 3), "deep": slice((2 * h) // 3, h)}


def audit_phase1(phase1_root: str | Path, protocol: str | Path, output: str | Path, edge_percentiles: list[float], tolerance_pixels: int, dx: float, dz: float) -> dict[str, Any]:
    phase1_root = Path(phase1_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_protocol(protocol)
    source, target = load_protocol_bundles(manifest)
    if target is None:
        raise ValueError("Phase-1b audit requires a cross-family target in the protocol.")
    thresholds = _source_thresholds(source.velocities, manifest.train_indices, edge_percentiles, dx, dz)
    (output / "edge_mask_thresholds.json").write_text(
        json.dumps({"mask_mode": "source_threshold", "thresholds": thresholds, "dx": dx, "dz": dz, "gradient_unit": "m/s per grid_cell"}, indent=2),
        encoding="utf-8",
    )

    variants = ["B1_raw_unet", "B2_hybrid_unet", "B3_raw_bed", "B4_pasd_fwi"]
    seeds = [0, 1, 2]
    coverage_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    depth_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    branch_summary: dict[str, Any] = {}

    val_pred_thresholds: dict[str, float] = {}
    for variant in variants:
        for seed in seeds:
            run_dir = phase1_root / variant / f"seed_{seed}"
            summary = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
            val = _predict_split(run_dir, source, manifest.val_indices, variant, summary)
            val_grad = gradient_magnitude_np(val["prediction"], dx=dx, dz=dz).reshape(-1)
            # Prediction threshold is calibrated only from FlatVel-A source validation predictions.
            val_pred_thresholds[f"{variant}_seed{seed}"] = float(np.percentile(val_grad, 90.0))

    for variant in variants:
        for seed in seeds:
            archive = _load_archive(phase1_root / variant / f"seed_{seed}" / "predictions_cross_family.npz")
            pred_threshold = val_pred_thresholds[f"{variant}_seed{seed}"]
            for row_index, sample_id in enumerate(archive["sample_id"].astype(int).tolist()):
                prediction = archive["prediction"][row_index]
                target_v = archive["target"][row_index]
                for percentile in edge_percentiles:
                    source_tau = thresholds[f"tau_source_{int(percentile)}"]
                    for mask_mode in ("source_threshold", "per_sample_percentile_diagnostic"):
                        if mask_mode == "source_threshold":
                            mask = edge_mask_from_threshold(target_v, source_tau, dx=dx, dz=dz)
                            edge_threshold = source_tau
                        else:
                            mask, edge_threshold = edge_mask_percentile(target_v, percentile, dx=dx, dz=dz)
                        identity = masked_mae_identity(prediction, target_v, mask)
                        base = {
                            "sample_id": sample_id,
                            "variant": variant,
                            "seed": seed,
                            "edge_percentile": int(percentile),
                            "mask_mode": mask_mode,
                            "edge_threshold": edge_threshold,
                            "identity_status": "PASS" if identity["weighted_identity_error"] <= 1e-5 else "FAILED",
                        }
                        coverage_rows.append({**base, **identity})
                        grad = gradient_metrics(prediction, target_v, mask, dx=dx, dz=dz)
                        pred_mask = gradient_magnitude_np(prediction, dx=dx, dz=dz) >= pred_threshold
                        prf = edge_prf(pred_mask, mask, tolerance_pixels=tolerance_pixels)
                        edge_rows.append({**base, **identity, **grad, **prf, "prediction_edge_threshold_source": "FlatVel-A source validation"})
                        if mask_mode == "source_threshold" and int(percentile) == 90:
                            gradient_rows.append({**base, **grad, **prf, "phase1_gradient_error_definition": "gradient_magnitude_MAE from PASD v0.2 metrics"})
                            h = target_v.shape[0]
                            zones = _depth_zone_slices(h)
                            for zone, sl in zones.items():
                                zmask = mask[sl]
                                zone_metrics = simple_metrics(prediction[sl], target_v[sl], zmask)
                                zone_grad = gradient_metrics(prediction[sl], target_v[sl], zmask, dx=dx, dz=dz)
                                zone_prf = edge_prf(pred_mask[sl], zmask, tolerance_pixels=tolerance_pixels)
                                depth_rows.append({
                                    "sample_id": sample_id,
                                    "variant": variant,
                                    "seed": seed,
                                    "zone": zone,
                                    **zone_metrics,
                                    **zone_grad,
                                    **zone_prf,
                                })

            if variant in {"B3_raw_bed", "B4_pasd_fwi"}:
                summary = json.loads((phase1_root / variant / f"seed_{seed}" / "metrics_summary.json").read_text(encoding="utf-8"))
                full = _predict_split(phase1_root / variant / f"seed_{seed}", target, manifest.cross_family_test_indices, variant, summary)
                closure = []
                for row_index, sample_id in enumerate(full["sample_id"].astype(int).tolist()):
                    target_v = full["target"][row_index]
                    pred_v = full["prediction"][row_index]
                    pred_bg = full["background"][row_index]
                    pred_res = full["residual"][row_index]
                    true_bg = gaussian_smooth_np(target_v, sigma=float(summary["training"].get("background_sigma", 1.5)))
                    true_res = target_v - true_bg
                    closure_error = float(np.abs((pred_bg + pred_res) - pred_v).mean())
                    closure.append(closure_error)
                    branch_rows.append({
                        "sample_id": sample_id,
                        "variant": variant,
                        "seed": seed,
                        "background_MAE": float(np.abs(pred_bg - true_bg).mean()),
                        "residual_MAE": float(np.abs(pred_res - true_res).mean()),
                        "background_SSIM": global_ssim_np(pred_bg, true_bg),
                        "residual_SSIM": global_ssim_np(pred_res, true_res),
                        "background_gradient_error": gradient_metrics(pred_bg, true_bg, dx=dx, dz=dz)["gradient_l1_all"],
                        "residual_gradient_error": gradient_metrics(pred_res, true_res, dx=dx, dz=dz)["gradient_l1_all"],
                        "pred_background_std": float(pred_bg.std()),
                        "pred_residual_std": float(pred_res.std()),
                        "true_background_std": float(true_bg.std()),
                        "true_residual_std": float(true_res.std()),
                        "branch_closure_mean_abs_error": closure_error,
                    })
                branch_summary[f"{variant}_seed{seed}"] = {
                    "mean_closure_error": float(np.mean(closure)),
                    "max_closure_error": float(np.max(closure)),
                    "notes": "Branch tensors were recomputed from Phase-1 checkpoints; predictions archives were not modified.",
                }

    write_csv(output / "edge_mask_coverage.csv", coverage_rows)
    write_csv(output / "edge_vs_nonedge_metrics.csv", edge_rows)
    write_csv(output / "gradient_metric_audit.csv", gradient_rows)
    write_csv(output / "background_residual_metrics.csv", branch_rows)
    write_csv(output / "per_depth_metrics.csv", depth_rows)
    audit = {
        "status": "PASS" if all(float(row["weighted_identity_error"]) <= 1e-5 for row in coverage_rows) else "FAIL",
        "gradient_unit": "m/s per grid_cell",
        "dx": dx,
        "dz": dz,
        "edge_percentiles": edge_percentiles,
        "prediction_threshold_calibration": "FlatVel-A source validation only",
        "phase1_gradient_error_difference_explanation": "Phase-1 gradient_error equals gradient magnitude MAE on physical predictions; Phase-1b adds component-wise L1, edge-masked L1, and direction cosine error.",
        "source_validation_prediction_thresholds": val_pred_thresholds,
    }
    (output / "gradient_metric_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "branch_diagnostics.json").write_text(json.dumps(branch_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    barplot(coverage_rows, x="variant", y="edge_coverage", group="mask_mode", output=output / "Figure_edge_mask_coverage.png", title="Edge mask coverage")
    boxplot({v: [float(r["edge_MAE"]) for r in edge_rows if r["variant"] == v and r["mask_mode"] == "source_threshold"] for v in variants}, output / "Figure_edge_vs_nonedge_MAE.png", "Source-threshold edge MAE", "edge_MAE")
    boxplot({v: [float(r["gradient_l1_edge"]) for r in gradient_rows if r["variant"] == v] for v in variants}, output / "Figure_gradient_metric_audit.png", "Gradient L1 on source-threshold edges", "gradient_l1_edge")
    boxplot({v: [float(r["residual_MAE"]) for r in branch_rows if r["variant"] == v] for v in ["B3_raw_bed", "B4_pasd_fwi"]}, output / "Figure_background_residual_diagnostics.png", "Residual branch diagnostics", "residual_MAE")
    boxplot({z: [float(r["gradient_l1_edge"]) for r in depth_rows if r["zone"] == z] for z in ["shallow", "middle", "deep"]}, output / "Figure_depthwise_metrics.png", "Depthwise gradient L1 edge", "gradient_l1_edge")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PASD Phase-1 metrics.")
    parser.add_argument("--phase1-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dx", default="auto")
    parser.add_argument("--dz", default="auto")
    parser.add_argument("--edge-percentiles", nargs="+", type=float, default=[85, 90])
    parser.add_argument("--tolerance-pixels", type=int, default=1)
    args = parser.parse_args()
    dx = 1.0 if args.dx == "auto" else float(args.dx)
    dz = 1.0 if args.dz == "auto" else float(args.dz)
    result = audit_phase1(args.phase1_root, args.protocol, args.output, args.edge_percentiles, args.tolerance_pixels, dx, dz)
    print(json.dumps({"status": result["status"], "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
