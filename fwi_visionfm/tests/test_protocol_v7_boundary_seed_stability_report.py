from __future__ import annotations

import csv
from pathlib import Path


def test_seed_stability_report_handles_missing_runs_and_guardrails(tmp_path: Path):
    from fwi_visionfm.scripts.report_protocol_v7_boundary_auxiliary_seed_stability import write_seed_stability_report

    summary_path = tmp_path / "protocol_v7_boundary_auxiliary_seed_stability_summary.csv"
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
            "val_SSIM": "0.8",
            "cross_family_MAE": "1.1",
            "cross_family_RMSE": "2.1",
            "cross_family_SSIM": "0.7",
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
            "val_SSIM": "0.81",
            "cross_family_MAE": "1.0",
            "cross_family_RMSE": "2.0",
            "cross_family_SSIM": "0.71",
            "cross_family_gradient_error": "0.28",
            "cross_family_edge_MAE": "0.39",
            "boundary_val_L1": "0.2",
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
            "val_MAE": "",
            "val_RMSE": "",
            "val_SSIM": "",
            "cross_family_MAE": "",
            "cross_family_RMSE": "",
            "cross_family_SSIM": "",
            "cross_family_gradient_error": "",
            "cross_family_edge_MAE": "",
            "boundary_val_L1": "",
            "status": "FAILED",
            "skip_reason": "RuntimeError",
        },
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report_path = write_seed_stability_report(tmp_path)
    text = report_path.read_text(encoding="utf-8")
    assert "not benchmark-level proof" in text
    assert "boundary auxiliary improves FWI generalization" not in text
    assert "FAILED" in text
    assert "majority-supported structural benefit" in text
