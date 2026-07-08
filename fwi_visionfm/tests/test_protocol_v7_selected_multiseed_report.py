from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image


def _write_png(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 18), color).save(path)


def test_protocol_v7_selected_multiseed_report_outputs(tmp_path: Path):
    from fwi_visionfm.scripts.report_protocol_v7_selected_multiseed import write_protocol_v7_selected_multiseed_report

    root = tmp_path / "seed_stability"
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "protocol_v7_boundary_auxiliary_seed_stability_summary.csv"
    rows = [
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
            "lambda_boundary": "0.10",
            "boundary_method": "gradient_magnitude",
            "val_MAE": "0.9",
            "val_RMSE": "1.9",
            "val_SSIM": "0.79",
            "cross_family_MAE": "1.0",
            "cross_family_RMSE": "2.0",
            "cross_family_SSIM": "0.69",
            "cross_family_gradient_error": "0.28",
            "cross_family_edge_MAE": "0.39",
            "boundary_val_L1": "0.20",
            "status": "SUCCESS",
            "skip_reason": "",
        },
        {
            "seed": "1",
            "model_type": "baseline",
            "decoder": "unet_decoder",
            "loss": "default_l1",
            "lambda_boundary": "",
            "boundary_method": "",
            "val_MAE": "1.2",
            "val_RMSE": "2.2",
            "val_SSIM": "0.81",
            "cross_family_MAE": "1.3",
            "cross_family_RMSE": "2.3",
            "cross_family_SSIM": "0.74",
            "cross_family_gradient_error": "0.33",
            "cross_family_edge_MAE": "0.43",
            "boundary_val_L1": "",
            "status": "SUCCESS",
            "skip_reason": "",
        },
        {
            "seed": "1",
            "model_type": "boundary_aux",
            "decoder": "boundary_aux_unet",
            "loss": "boundary_aux_l1",
            "lambda_boundary": "0.10",
            "boundary_method": "gradient_magnitude",
            "val_MAE": "1.1",
            "val_RMSE": "2.1",
            "val_SSIM": "0.73",
            "cross_family_MAE": "1.2",
            "cross_family_RMSE": "2.2",
            "cross_family_SSIM": "0.72",
            "cross_family_gradient_error": "0.31",
            "cross_family_edge_MAE": "0.42",
            "boundary_val_L1": "0.21",
            "status": "SUCCESS",
            "skip_reason": "",
        },
        {
            "seed": "2",
            "model_type": "baseline",
            "decoder": "unet_decoder",
            "loss": "default_l1",
            "lambda_boundary": "",
            "boundary_method": "",
            "val_MAE": "1.4",
            "val_RMSE": "2.4",
            "val_SSIM": "0.70",
            "cross_family_MAE": "1.5",
            "cross_family_RMSE": "2.5",
            "cross_family_SSIM": "0.67",
            "cross_family_gradient_error": "0.36",
            "cross_family_edge_MAE": "0.46",
            "boundary_val_L1": "",
            "status": "SUCCESS",
            "skip_reason": "",
        },
        {
            "seed": "2",
            "model_type": "boundary_aux",
            "decoder": "boundary_aux_unet",
            "loss": "boundary_aux_l1",
            "lambda_boundary": "0.10",
            "boundary_method": "gradient_magnitude",
            "val_MAE": "1.3",
            "val_RMSE": "2.3",
            "val_SSIM": "0.69",
            "cross_family_MAE": "1.4",
            "cross_family_RMSE": "2.4",
            "cross_family_SSIM": "0.68",
            "cross_family_gradient_error": "0.34",
            "cross_family_edge_MAE": "0.44",
            "boundary_val_L1": "0.22",
            "status": "SUCCESS",
            "skip_reason": "",
        },
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (root / "protocol_v7_boundary_auxiliary_seed_stability_report.md").write_text("seed stability stub", encoding="utf-8")

    for seed in ("0", "1", "2"):
        for model_type, color in (("baseline", (20, 40, 60)), ("boundary_aux", (120, 60, 20))):
            run_dir = root / f"seed_{seed}_{model_type}"
            _write_png(run_dir / "prediction_grid.png", color)
            _write_png(run_dir / "gradient_grid.png", color)
            if model_type == "boundary_aux":
                _write_png(run_dir / "boundary_prediction_grid.png", color)
                _write_png(run_dir / "boundary_target_grid.png", color)

    output_dir = tmp_path / "stage_reports" / "protocol_v7_selected_multiseed"
    result = write_protocol_v7_selected_multiseed_report(root, output_dir)

    report_text = (result["report_path"]).read_text(encoding="utf-8")
    claims_text = (result["claims_path"]).read_text(encoding="utf-8")
    assert "gradient_error: 3/3" in report_text
    assert "edge_MAE: 3/3" in report_text
    assert "SSIM improvement is not stable" in report_text
    assert "not benchmark-level proof" in report_text
    assert "boundary auxiliary improves FWI generalization" not in report_text
    assert "## 可以写" in claims_text
    assert "## 不能写" in claims_text
    assert result["summary_path"].exists()
    assert result["metrics_bar_path"].exists()
    assert result["win_count_bar_path"].exists()
    assert result["prediction_grid_path"].exists()
    assert result["gradient_grid_path"].exists()
    assert result["boundary_grid_path"].exists()
