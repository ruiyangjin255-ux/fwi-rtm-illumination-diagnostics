from __future__ import annotations

import csv
from pathlib import Path


def test_boundary_tuning_report_contains_reuse_and_guardrails(tmp_path: Path):
    from fwi_visionfm.scripts.report_protocol_v7_boundary_auxiliary_tuning import write_protocol_v7_boundary_auxiliary_tuning_report

    root = tmp_path / "tuning"
    root.mkdir(parents=True, exist_ok=True)
    summary_rows = [
        {
            "run_id": "baseline_seed0",
            "seed": "0",
            "lambda_boundary": "",
            "boundary_method": "",
            "threshold": "",
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
            "reused_from": "seed_stability:baseline",
            "skip_reason": "",
        },
        {
            "run_id": "lambda010_seed0",
            "seed": "0",
            "lambda_boundary": "0.1",
            "boundary_method": "gradient_magnitude",
            "threshold": "",
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
            "reused_from": "seed_stability:boundary_aux",
            "skip_reason": "",
        },
        {
            "run_id": "thresholded_seed0",
            "seed": "0",
            "lambda_boundary": "0.05",
            "boundary_method": "thresholded_gradient",
            "threshold": "0.2",
            "val_MAE": "0.88",
            "val_RMSE": "1.88",
            "val_SSIM": "0.76",
            "cross_family_MAE": "0.98",
            "cross_family_RMSE": "1.98",
            "cross_family_SSIM": "0.75",
            "cross_family_gradient_error": "0.27",
            "cross_family_edge_MAE": "0.37",
            "boundary_val_L1": "0.18",
            "status": "SUCCESS",
            "reused_from": "",
            "skip_reason": "",
        },
    ]
    summary_path = root / "protocol_v7_boundary_auxiliary_tuning_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    report_path = write_protocol_v7_boundary_auxiliary_tuning_report(root)
    text = report_path.read_text(encoding="utf-8")
    assert "Reused Results" in text
    assert "not benchmark-level proof" in text
    assert "boundary auxiliary improves FWI generalization" not in text
    assert "thresholded_gradient" in text
