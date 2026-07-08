from __future__ import annotations

import csv
from pathlib import Path


def test_protocol_v9_selected_comparison_report_contains_required_constraints(tmp_path: Path):
    from fwi_visionfm.scripts.report_protocol_v9_selected_comparison import write_protocol_v9_selected_comparison_report

    summary_rows = [
        {"method_name": "cnn_baseline_unet_l1", "method_family": "task_specific_supervised", "seed": 0, "backbone": "cnn_baseline", "feature_type": "end_to_end_supervised", "decoder": "unet_decoder", "loss": "default_l1", "is_real_feature": False, "train_size": 100, "val_size": 50, "test_size": 50, "cross_family_MAE": 430.0, "cross_family_RMSE": 525.0, "cross_family_SSIM": 0.81, "cross_family_gradient_error": 41.0, "cross_family_edge_MAE": 112.0, "status": "SUCCESS", "reused_from": "", "limitation_note": ""},
        {"method_name": "boundary_aux_gradient_lambda010", "method_family": "boundary_auxiliary", "seed": 0, "backbone": "cnn_baseline", "feature_type": "end_to_end_supervised", "decoder": "boundary_aux_unet", "loss": "boundary_aux_l1", "is_real_feature": False, "train_size": 100, "val_size": 50, "test_size": 50, "cross_family_MAE": 429.0, "cross_family_RMSE": 521.0, "cross_family_SSIM": 0.80, "cross_family_gradient_error": 40.0, "cross_family_edge_MAE": 111.0, "status": "SUCCESS", "reused_from": "", "limitation_note": ""},
        {"method_name": "vit_mae_base_frozen_decoder", "method_family": "natural_image_mae_frozen", "seed": 0, "backbone": "vit_mae_base", "feature_type": "frozen_decoder_only", "decoder": "lightweight_feature_decoder", "loss": "default_l1", "is_real_feature": True, "train_size": 100, "val_size": 50, "test_size": 50, "cross_family_MAE": 422.0, "cross_family_RMSE": 522.0, "cross_family_SSIM": 0.82, "cross_family_gradient_error": 61.0, "cross_family_edge_MAE": 132.0, "status": "SUCCESS", "reused_from": "", "limitation_note": ""},
        {"method_name": "ncs2d_frozen_decoder", "method_family": "seismic_domain_ncs_frozen", "seed": 0, "backbone": "ncs_2d", "feature_type": "frozen_decoder_only", "decoder": "lightweight_feature_decoder", "loss": "default_l1", "is_real_feature": True, "train_size": 100, "val_size": 50, "test_size": 50, "cross_family_MAE": 419.0, "cross_family_RMSE": 519.0, "cross_family_SSIM": 0.83, "cross_family_gradient_error": 59.0, "cross_family_edge_MAE": 129.0, "status": "SUCCESS", "reused_from": "", "limitation_note": ""},
    ]
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = write_protocol_v9_selected_comparison_report(rows=summary_rows, output_dir=out_dir)
    report_text = payload["report_path"].read_text(encoding="utf-8")
    claims_text = payload["claims_path"].read_text(encoding="utf-8")
    assert "ncs2d_frozen_decoder" in report_text
    assert "vit_mae_base_frozen_decoder" in report_text
    assert "boundary_aux_gradient_lambda010" in report_text
    assert "selected comparison" in report_text
    assert "not benchmark-level proof" in report_text
    assert "methods are not perfectly matched" in report_text
    assert "NCS improves FWI" not in report_text
    assert "NCS outperforms CNN" not in report_text
    assert "## Can Claim" in claims_text
    assert "## Cannot Claim" in claims_text
