from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.evaluation.metrics import compute_velocity_metrics
from fwi_visionfm.evaluation.prediction_alignment import align_predictions_by_sample_id, load_prediction_npz, validate_prediction_targets
from fwi_visionfm.scripts.run_protocol_v4_integrated_visual_search import _write_triplet_grid


ALPHAS = [round(i * 0.1, 1) for i in range(11)]
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]


def _blur(batch: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    pad = kernel_size // 2
    padded = np.pad(batch, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    out = np.zeros_like(batch, dtype=np.float32)
    for i in range(kernel_size):
        for j in range(kernel_size):
            out += padded[:, i : i + batch.shape[1], j : j + batch.shape[2]]
    return out / float(kernel_size * kernel_size)


def _gradient_mag(batch: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(batch.astype(np.float32), axis=(1, 2))
    return np.sqrt(gx * gx + gy * gy)


def average_fusion(pred_a: np.ndarray, pred_b: np.ndarray, alpha: float) -> np.ndarray:
    return (float(alpha) * pred_a + (1.0 - float(alpha)) * pred_b).astype(np.float32)


def low_high_fusion(pred_a: np.ndarray, pred_b: np.ndarray, beta: float) -> np.ndarray:
    low_a = _blur(pred_a)
    high_b = pred_b - _blur(pred_b)
    return (low_a + float(beta) * high_b).astype(np.float32)


def edge_aware_fusion(pred_a: np.ndarray, pred_b: np.ndarray, edge_scale: float) -> np.ndarray:
    grad = _gradient_mag(pred_b)
    denom = np.quantile(grad, 0.9)
    weights = np.clip(grad / max(float(denom), 1.0e-6) * float(edge_scale), 0.0, 1.0)
    return ((1.0 - weights) * pred_a + weights * pred_b).astype(np.float32)


def _load_prediction(path: Path) -> dict[str, Any]:
    return load_prediction_npz(path)


def _split_path(run_dir: Path, requested: str) -> tuple[Path, str]:
    candidates = []
    if requested == "val":
        candidates = [("val", run_dir / "predictions_val.npz"), ("in_family_test", run_dir / "predictions_in_family_test.npz")]
    elif requested in {"in_family", "in_family_test"}:
        candidates = [("in_family_test", run_dir / "predictions_in_family_test.npz")]
    else:
        candidates = [(requested, run_dir / f"predictions_{requested}.npz")]
    for name, path in candidates:
        if path.exists():
            return path, name
    raise FileNotFoundError(f"no prediction npz for {requested} under {run_dir}")


def _apply(method: str, pred_a: np.ndarray, pred_b: np.ndarray, param: float) -> np.ndarray:
    if method == "average_fusion":
        return average_fusion(pred_a, pred_b, alpha=param)
    if method == "low_high_fusion":
        return low_high_fusion(pred_a, pred_b, beta=param)
    if method == "edge_aware_fusion":
        return edge_aware_fusion(pred_a, pred_b, edge_scale=param)
    raise ValueError(f"unsupported fusion method: {method}")


def _grid_params(method: str) -> list[float]:
    return ALPHAS if method == "average_fusion" else BETAS


def _metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "MAE": metrics["mae"],
        "RMSE": metrics["rmse"],
        "SSIM": metrics["ssim"],
        "gradient_error": metrics["gradient_error"],
        "edge_MAE": metrics["edge_mae"],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_pair_fusion(
    *,
    output_dir: str | Path,
    run_a: str | Path,
    run_b: str | Path,
    method: str,
    optimize_on: str,
    reference_only: bool,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    run_a = Path(run_a)
    run_b = Path(run_b)
    opt_a_path, opt_actual = _split_path(run_a, optimize_on)
    opt_b_path, opt_actual_b = _split_path(run_b, optimize_on)
    if opt_actual_b != opt_actual:
        raise ValueError(f"optimization split mismatch: {opt_actual} vs {opt_actual_b}")
    opt_a = _load_prediction(opt_a_path)
    opt_b = _load_prediction(opt_b_path)
    if opt_a.get("has_explicit_sample_id") and opt_b.get("has_explicit_sample_id"):
        aligned_opt = align_predictions_by_sample_id(opt_a, opt_b)
        if aligned_opt["status"] != "ALIGNED":
            raise ValueError(f"optimization alignment failed: {aligned_opt['status']}")
        opt_pred_a = aligned_opt["prediction_a"]
        opt_pred_b = aligned_opt["prediction_b"]
        opt_target = aligned_opt["target"]
        opt_sample_id = aligned_opt["sample_id"]
    else:
        check = validate_prediction_targets(opt_a, opt_b)
        if check["status"] != "MATCH":
            raise ValueError("optimization targets differ")
        if Path(run_a) != Path(run_b):
            raise ValueError("optimization sample_id unavailable for cross-run fusion fallback")
        opt_pred_a = np.asarray(opt_a["prediction"], dtype=np.float32)
        opt_pred_b = np.asarray(opt_b["prediction"], dtype=np.float32)
        opt_target = np.asarray(opt_a["target"], dtype=np.float32)
        opt_sample_id = list(opt_a["sample_id"])
    candidates = []
    for param in _grid_params(method):
        fused = _apply(method, opt_pred_a, opt_pred_b, param)
        metrics = compute_velocity_metrics(fused, opt_target)
        candidates.append({"param": float(param), **_metric_row(metrics)})
    best = min(candidates, key=lambda row: (row["MAE"] + row["RMSE"] + row["gradient_error"] + row["edge_MAE"]))
    best_param = float(best["param"])
    test_a = _load_prediction(run_a / "predictions_cross_family_test.npz")
    test_b = _load_prediction(run_b / "predictions_cross_family_test.npz")
    if test_a.get("has_explicit_sample_id") and test_b.get("has_explicit_sample_id"):
        aligned_test = align_predictions_by_sample_id(test_a, test_b)
        if aligned_test["status"] != "ALIGNED":
            raise ValueError(f"test alignment failed: {aligned_test['status']}")
        test_pred_a = aligned_test["prediction_a"]
        test_pred_b = aligned_test["prediction_b"]
        test_target = aligned_test["target"]
        test_sample_id = aligned_test["sample_id"]
    else:
        check = validate_prediction_targets(test_a, test_b)
        if check["status"] != "MATCH":
            raise ValueError("test targets differ")
        if Path(run_a) != Path(run_b):
            raise ValueError("test sample_id unavailable for cross-run fusion fallback")
        test_pred_a = np.asarray(test_a["prediction"], dtype=np.float32)
        test_pred_b = np.asarray(test_b["prediction"], dtype=np.float32)
        test_target = np.asarray(test_a["target"], dtype=np.float32)
        test_sample_id = list(test_a["sample_id"])
    fused_test = _apply(method, test_pred_a, test_pred_b, best_param)
    test_metrics = compute_velocity_metrics(fused_test, test_target)
    opt_fused = _apply(method, opt_pred_a, opt_pred_b, best_param)
    opt_metrics = compute_velocity_metrics(opt_fused, opt_target)
    np.savez(
        out / "fused_predictions_cross_family_test.npz",
        prediction=fused_test.astype(np.float32),
        target=test_target.astype(np.float32),
        velocity_pred_physical=fused_test.astype(np.float32),
        velocity_true_physical=test_target.astype(np.float32),
        error_map_physical=(fused_test - test_target).astype(np.float32),
        sample_id=np.asarray(test_sample_id),
        metric_space=np.asarray("physical_velocity"),
    )
    _write_json(out / "fused_metrics_val.json", {**opt_metrics, "metric_space": "physical_velocity"})
    _write_json(out / "fused_metrics_cross_family_test.json", {**test_metrics, "metric_space": "physical_velocity"})
    _write_json(
        out / "fusion_config.json",
        {
            "method": method,
            "run_a": str(run_a),
            "run_b": str(run_b),
            "best_param": best_param,
            "param_search": candidates,
            "optimize_requested": optimize_on,
            "optimize_actual": opt_actual,
            "optimize_sample_count": len(opt_sample_id),
            "cross_test_sample_count": len(test_sample_id),
            "reference_only": bool(reference_only),
            "status": "SUCCESS",
        },
    )
    _write_triplet_grid(out / "fused_predictions_cross_family_test.npz", out / "fused_prediction_grid.png", gradient=False)
    _write_triplet_grid(out / "fused_predictions_cross_family_test.npz", out / "fused_gradient_grid.png", gradient=True)
    return {"best_param": best_param, "optimize_actual": opt_actual, **_metric_row(test_metrics)}


def _run_path(root: Path, model: str, bridge: str, loss: str, seed: int) -> Path:
    return root / "flatvel_a_subset2k_to_curvevel_a_subset500" / model / bridge / "unet_decoder" / loss / f"seed_{seed}"


def run_all_fusions(root: str | Path, output_root: str | Path, methods: list[str], optimize_on: str) -> dict[str, Any]:
    root = Path(root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    numeric_runs = [
        ("cnn_res3_seed0", _run_path(root, "cnn_baseline", "raw_envelope_spectrum3", "default_l1", 0)),
        ("cnn_res3_seed1", _run_path(root, "cnn_baseline", "raw_envelope_spectrum3", "default_l1", 1)),
    ]
    structural_runs = [("vit_multiband_structure_seed0", _run_path(root, "vit_tiny_scratch", "spectrogram_multiband", "structure_loss", 0))]
    dino_runs = [
        ("dino_res3_seed0", _run_path(root, "dinov2_lora_smoke", "raw_envelope_spectrum3", "default_l1", 0)),
        ("dino_multiband_seed0", _run_path(root, "dinov2_lora_smoke", "spectrogram_multiband", "default_l1", 0)),
    ]
    rows = []
    def _record_skip(name: str, method: str, a_name: str, b_name: str, a_path: Path, b_path: Path, reference_only: bool, reason: str) -> dict[str, Any]:
        out = output_root / name
        out.mkdir(parents=True, exist_ok=True)
        _write_json(
            out / "fusion_config.json",
            {
                "method": method,
                "run_a": str(a_path),
                "run_b": str(b_path),
                "best_param": "",
                "optimize_requested": optimize_on,
                "optimize_actual": "",
                "reference_only": bool(reference_only),
                "status": "SKIPPED_TARGET_MISMATCH",
                "skip_reason": reason,
            },
        )
        return {
            "fusion_name": name,
            "method": method,
            "source_a": a_name,
            "source_b": b_name,
            "reference_only": reference_only,
            "status": "SKIPPED_TARGET_MISMATCH",
            "skip_reason": reason,
        }

    for a_name, a_path in numeric_runs:
        for b_name, b_path in structural_runs:
            for method in methods:
                name = f"{method}__{a_name}__{b_name}"
                try:
                    result = run_pair_fusion(output_dir=output_root / name, run_a=a_path, run_b=b_path, method=method, optimize_on=optimize_on, reference_only=False)
                    rows.append({"fusion_name": name, "method": method, "source_a": a_name, "source_b": b_name, "reference_only": False, "status": "SUCCESS", **result})
                except ValueError as exc:
                    rows.append(_record_skip(name, method, a_name, b_name, a_path, b_path, False, str(exc)))
    for a_name, a_path in dino_runs:
        for b_name, b_path in structural_runs:
            for method in methods:
                name = f"{method}__{a_name}__{b_name}"
                try:
                    result = run_pair_fusion(output_dir=output_root / name, run_a=a_path, run_b=b_path, method=method, optimize_on=optimize_on, reference_only=True)
                    rows.append({"fusion_name": name, "method": method, "source_a": a_name, "source_b": b_name, "reference_only": True, "status": "SUCCESS", **result})
                except ValueError as exc:
                    rows.append(_record_skip(name, method, a_name, b_name, a_path, b_path, True, str(exc)))
    _write_json(output_root / "fusion_run_summary.json", {"run_count": len(rows), "runs": rows})
    return {"run_count": len(rows), "runs": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse Protocol V4 predictions.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=["average_fusion", "low_high_fusion", "edge_aware_fusion"])
    parser.add_argument("--optimize-on", default="val")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_all_fusions(args.root, args.output_root, args.methods, args.optimize_on)
    print(f"Wrote {Path(args.output_root) / 'fusion_run_summary.json'}")
    print(f"run_count={summary['run_count']}")


if __name__ == "__main__":
    main()
