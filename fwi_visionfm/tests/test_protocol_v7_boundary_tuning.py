from __future__ import annotations

import csv
import json
from pathlib import Path


def _write_run(root: Path, *, name: str, config: dict, val: dict | None = None, cross: dict | None = None) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if val is not None:
        (run_dir / "metrics_val.json").write_text(json.dumps(val), encoding="utf-8")
    if cross is not None:
        (run_dir / "metrics_cross_family_test.json").write_text(json.dumps(cross), encoding="utf-8")
    return run_dir


def test_boundary_tuning_summary_handles_reuse_and_threshold(tmp_path: Path):
    from fwi_visionfm.scripts.run_protocol_v7_boundary_auxiliary_tuning import build_boundary_tuning_summary

    root = tmp_path / "tuning"
    reuse_seed_root = tmp_path / "seed_stability"
    reuse_smoke_root = tmp_path / "smoke"

    summary_rows = [
        {
            "seed": "0",
            "model_type": "baseline",
            "decoder": "unet_decoder",
            "loss": "default_l1",
            "lambda_boundary": "",
            "boundary_method": "",
            "val_MAE": "1.0",
            "val_RMSE": "2.0",
            "val_SSIM": "0.80",
            "cross_family_MAE": "1.1",
            "cross_family_RMSE": "2.1",
            "cross_family_SSIM": "0.70",
            "cross_family_gradient_error": "0.30",
            "cross_family_edge_MAE": "0.40",
            "boundary_val_L1": "",
            "status": "SUCCESS",
            "skip_reason": "",
        },
        {
            "seed": "0",
            "model_type": "boundary_aux",
            "decoder": "boundary_aux_unet",
            "loss": "boundary_aux_l1",
            "lambda_boundary": "0.1",
            "boundary_method": "gradient_magnitude",
            "val_MAE": "0.9",
            "val_RMSE": "1.9",
            "val_SSIM": "0.79",
            "cross_family_MAE": "1.0",
            "cross_family_RMSE": "2.0",
            "cross_family_SSIM": "0.69",
            "cross_family_gradient_error": "0.28",
            "cross_family_edge_MAE": "0.39",
            "boundary_val_L1": "0.2",
            "status": "SUCCESS",
            "skip_reason": "",
        },
    ]
    reuse_seed_root.mkdir(parents=True, exist_ok=True)
    with (reuse_seed_root / "protocol_v7_boundary_auxiliary_seed_stability_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    _write_run(
        reuse_smoke_root,
        name="run_2",
        config={
            "seed": 0,
            "decoder": "boundary_aux_unet",
            "loss": "boundary_aux_l1",
            "lambda_boundary": 0.05,
            "boundary_method": "gradient_magnitude",
            "status": "SUCCESS",
        },
        val={"mae": 1.2, "rmse": 2.2, "ssim": 0.78, "boundary_val_l1": 0.21},
        cross={"mae": 1.3, "rmse": 2.3, "ssim": 0.71, "gradient_error": 0.29, "edge_mae": 0.38},
    )
    _write_run(
        reuse_smoke_root,
        name="run_4",
        config={
            "seed": 0,
            "decoder": "boundary_aux_unet",
            "loss": "boundary_aux_l1",
            "lambda_boundary": 0.05,
            "boundary_method": "sobel",
            "status": "SUCCESS",
        },
        val={"mae": 1.25, "rmse": 2.25, "ssim": 0.80, "boundary_val_l1": 0.22},
        cross={"mae": 1.35, "rmse": 2.35, "ssim": 0.72, "gradient_error": 0.31, "edge_mae": 0.39},
    )

    _write_run(
        root,
        name="lambda003_seed1",
        config={
            "run_id": "lambda003_seed1",
            "seed": 1,
            "lambda_boundary": 0.03,
            "boundary_method": "gradient_magnitude",
            "threshold": None,
            "status": "SUCCESS",
            "reused_from": "",
        },
        val={"mae": 1.4, "rmse": 2.4, "ssim": 0.82, "boundary_val_l1": 0.19},
        cross={"mae": 1.5, "rmse": 2.5, "ssim": 0.73, "gradient_error": 0.27, "edge_mae": 0.36},
    )
    _write_run(
        root,
        name="thresholded_seed0",
        config={
            "run_id": "thresholded_seed0",
            "seed": 0,
            "lambda_boundary": 0.05,
            "boundary_method": "thresholded_gradient",
            "threshold": 0.2,
            "status": "SUCCESS",
            "reused_from": "",
        },
        val={"mae": 1.45, "rmse": 2.45, "ssim": 0.74, "boundary_val_l1": 0.18},
        cross={"mae": 1.55, "rmse": 2.55, "ssim": 0.75, "gradient_error": 0.26, "edge_mae": 0.35},
    )

    summary_path = build_boundary_tuning_summary(root=root, reuse_seed_stability_root=reuse_seed_root, reuse_smoke_root=reuse_smoke_root)
    rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8")))
    assert any(row["reused_from"] for row in rows)
    assert any(row["boundary_method"] == "thresholded_gradient" and row["threshold"] == "0.2" for row in rows)
    assert any(row["lambda_boundary"] == "0.1" and row["reused_from"] for row in rows)

