from __future__ import annotations

import csv
from pathlib import Path


def test_protocol_v7_fake_summary_generates_report(tmp_path: Path):
    from fwi_visionfm.scripts.report_protocol_v7_boundary_auxiliary import write_protocol_v7_boundary_auxiliary_report

    summary_path = tmp_path / "protocol_v7_boundary_auxiliary_summary.csv"
    rows = [
        {
            "run_id": "run_1",
            "decoder": "unet_decoder",
            "loss": "default_l1",
            "lambda_boundary": "",
            "boundary_method": "",
            "seed": "0",
            "train_size": "100",
            "val_size": "50",
            "test_size": "50",
            "epochs": "2",
            "val_MAE": "1.0",
            "val_RMSE": "2.0",
            "val_SSIM": "0.8",
            "cross_family_MAE": "1.1",
            "cross_family_RMSE": "2.1",
            "cross_family_SSIM": "0.7",
            "cross_family_gradient_error": "0.3",
            "cross_family_edge_MAE": "0.4",
            "boundary_val_L1": "",
            "status": "SUCCESS",
            "skip_reason": "",
        },
        {
            "run_id": "run_2",
            "decoder": "boundary_aux_unet",
            "loss": "boundary_aux_l1",
            "lambda_boundary": "0.05",
            "boundary_method": "gradient_magnitude",
            "seed": "0",
            "train_size": "100",
            "val_size": "50",
            "test_size": "50",
            "epochs": "2",
            "val_MAE": "1.2",
            "val_RMSE": "2.2",
            "val_SSIM": "0.75",
            "cross_family_MAE": "1.3",
            "cross_family_RMSE": "2.3",
            "cross_family_SSIM": "0.72",
            "cross_family_gradient_error": "0.31",
            "cross_family_edge_MAE": "0.41",
            "boundary_val_L1": "0.2",
            "status": "SUCCESS",
            "skip_reason": "",
        },
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    report = write_protocol_v7_boundary_auxiliary_report(tmp_path)
    text = report.read_text(encoding="utf-8")
    assert "boundary auxiliary smoke" in text.lower()
    assert "not benchmark-level proof" in text
    assert "boundary auxiliary improves FWI generalization" not in text
