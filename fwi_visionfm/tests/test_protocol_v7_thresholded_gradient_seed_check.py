from __future__ import annotations

import csv
from pathlib import Path


def test_thresholded_seed_check_summary_handles_reuse(tmp_path: Path):
    from fwi_visionfm.scripts.run_protocol_v7_thresholded_gradient_seed_check import build_thresholded_seed_check_summary

    root = tmp_path / "seed_check"
    reuse_tuning = tmp_path / "tuning"
    reuse_tuning.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "run_id": "thresholded_seed0",
            "seed": "0",
            "lambda_boundary": "0.05",
            "boundary_method": "thresholded_gradient",
            "threshold": "0.2",
            "val_MAE": "1.0",
            "val_RMSE": "2.0",
            "val_SSIM": "0.8",
            "cross_family_MAE": "1.1",
            "cross_family_RMSE": "2.1",
            "cross_family_SSIM": "0.7",
            "cross_family_gradient_error": "0.3",
            "cross_family_edge_MAE": "0.4",
            "boundary_val_L1": "0.2",
            "status": "SUCCESS",
            "reused_from": "",
            "skip_reason": "",
        },
        {
            "run_id": "lambda005_seed0",
            "seed": "0",
            "lambda_boundary": "0.05",
            "boundary_method": "gradient_magnitude",
            "threshold": "",
            "val_MAE": "1.2",
            "val_RMSE": "2.2",
            "val_SSIM": "0.81",
            "cross_family_MAE": "1.3",
            "cross_family_RMSE": "2.3",
            "cross_family_SSIM": "0.71",
            "cross_family_gradient_error": "0.31",
            "cross_family_edge_MAE": "0.41",
            "boundary_val_L1": "0.21",
            "status": "SUCCESS",
            "reused_from": "smoke:run_2",
            "skip_reason": "",
        },
        {
            "run_id": "lambda010_seed0",
            "seed": "0",
            "lambda_boundary": "0.1",
            "boundary_method": "gradient_magnitude",
            "threshold": "",
            "val_MAE": "1.4",
            "val_RMSE": "2.4",
            "val_SSIM": "0.79",
            "cross_family_MAE": "1.5",
            "cross_family_RMSE": "2.5",
            "cross_family_SSIM": "0.69",
            "cross_family_gradient_error": "0.29",
            "cross_family_edge_MAE": "0.39",
            "boundary_val_L1": "0.22",
            "status": "SUCCESS",
            "reused_from": "seed_stability:boundary_aux_seed0",
            "skip_reason": "",
        },
    ]
    with (reuse_tuning / "protocol_v7_boundary_auxiliary_tuning_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = build_thresholded_seed_check_summary(root=root, reuse_tuning_root=reuse_tuning)
    result_rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8")))
    assert any(row["reused_from"] for row in result_rows)
    assert any(row["boundary_method"] == "thresholded_gradient" for row in result_rows)

