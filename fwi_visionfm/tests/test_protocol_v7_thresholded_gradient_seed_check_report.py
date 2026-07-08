from __future__ import annotations

import csv
from pathlib import Path


def test_thresholded_seed_check_report_contains_guardrails(tmp_path: Path):
    from fwi_visionfm.scripts.report_protocol_v7_thresholded_gradient_seed_check import write_protocol_v7_thresholded_gradient_seed_check_report

    root = tmp_path / "seed_check"
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "seed": "0",
            "boundary_method": "thresholded_gradient",
            "lambda_boundary": "0.05",
            "threshold": "0.2",
            "MAE": "1.1",
            "RMSE": "2.1",
            "SSIM": "0.7",
            "gradient_error": "0.3",
            "edge_MAE": "0.4",
            "status": "SUCCESS",
            "reused_from": "tuning:thresholded_seed0",
            "skip_reason": "",
        },
        {
            "seed": "0",
            "boundary_method": "gradient_magnitude_lambda005",
            "lambda_boundary": "0.05",
            "threshold": "",
            "MAE": "1.3",
            "RMSE": "2.3",
            "SSIM": "0.71",
            "gradient_error": "0.31",
            "edge_MAE": "0.41",
            "status": "SUCCESS",
            "reused_from": "tuning:lambda005_seed0",
            "skip_reason": "",
        },
        {
            "seed": "0",
            "boundary_method": "gradient_magnitude_lambda010",
            "lambda_boundary": "0.1",
            "threshold": "",
            "MAE": "1.5",
            "RMSE": "2.5",
            "SSIM": "0.69",
            "gradient_error": "0.29",
            "edge_MAE": "0.39",
            "status": "SUCCESS",
            "reused_from": "tuning:lambda010_seed0",
            "skip_reason": "",
        },
    ]
    with (root / "protocol_v7_thresholded_gradient_seed_check_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = write_protocol_v7_thresholded_gradient_seed_check_report(root)
    text = report.read_text(encoding="utf-8")
    assert "thresholded_gradient" in text
    assert "not benchmark-level proof" in text
    assert "boundary auxiliary improves FWI generalization" not in text
    assert "reused_from" in text.lower() or "reused" in text.lower()
