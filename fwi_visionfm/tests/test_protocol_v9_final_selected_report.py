from __future__ import annotations

from pathlib import Path


def test_protocol_v9_final_selected_report_generation(tmp_path: Path):
    from fwi_visionfm.scripts.report_protocol_v9_final_selected import write_protocol_v9_final_selected_report

    selected_summary = tmp_path / "selected.csv"
    selected_summary.write_text(
        "\n".join(
            [
                "method_name,method_family,seed,backbone,feature_type,decoder,loss,is_real_feature,train_size,val_size,test_size,cross_family_MAE,cross_family_RMSE,cross_family_SSIM,cross_family_gradient_error,cross_family_edge_MAE,status,reused_from,limitation_note",
                "cnn_baseline_unet_l1,task_specific_supervised,0,cnn_baseline,end_to_end_supervised,unet_decoder,default_l1,False,100,50,50,430,525,0.81,41,112,SUCCESS,,baseline",
                "cnn_baseline_unet_l1,task_specific_supervised,1,cnn_baseline,end_to_end_supervised,unet_decoder,default_l1,False,100,50,50,431,526,0.82,42,113,SUCCESS,,baseline",
                "cnn_baseline_unet_l1,task_specific_supervised,2,cnn_baseline,end_to_end_supervised,unet_decoder,default_l1,False,100,50,50,432,527,0.83,43,114,SUCCESS,,baseline",
                "boundary_aux_gradient_lambda010,boundary_auxiliary,0,cnn_baseline,end_to_end_supervised,boundary_aux_unet,boundary_aux_l1,False,100,50,50,429,521,0.80,40,111,SUCCESS,,boundary",
                "boundary_aux_gradient_lambda010,boundary_auxiliary,1,cnn_baseline,end_to_end_supervised,boundary_aux_unet,boundary_aux_l1,False,100,50,50,428,520,0.81,41,112,SUCCESS,,boundary",
                "boundary_aux_gradient_lambda010,boundary_auxiliary,2,cnn_baseline,end_to_end_supervised,boundary_aux_unet,boundary_aux_l1,False,100,50,50,427,519,0.82,42,113,SUCCESS,,boundary",
                "vit_mae_base_frozen_decoder,natural_image_mae_frozen,0,vit_mae_base,frozen_decoder_only,lightweight_feature_decoder,default_l1,True,100,50,50,423,520,0.83,56,134,SUCCESS,,vit",
                "vit_mae_base_frozen_decoder,natural_image_mae_frozen,1,vit_mae_base,frozen_decoder_only,lightweight_feature_decoder,default_l1,True,100,50,50,424,521,0.83,55,133,SUCCESS,,vit",
                "vit_mae_base_frozen_decoder,natural_image_mae_frozen,2,vit_mae_base,frozen_decoder_only,lightweight_feature_decoder,default_l1,True,100,50,50,422,519,0.83,56,134,SUCCESS,,vit",
                "ncs2d_frozen_decoder,seismic_domain_ncs_frozen,0,ncs_2d,frozen_decoder_only,lightweight_feature_decoder,default_l1,True,100,50,50,420,519,0.83,60,131,SUCCESS,,ncs",
                "ncs2d_frozen_decoder,seismic_domain_ncs_frozen,1,ncs_2d,frozen_decoder_only,lightweight_feature_decoder,default_l1,True,100,50,50,419,518,0.83,61,132,SUCCESS,,ncs",
                "ncs2d_frozen_decoder,seismic_domain_ncs_frozen,2,ncs_2d,frozen_decoder_only,lightweight_feature_decoder,default_l1,True,100,50,50,418,517,0.83,59,131,SUCCESS,,ncs",
            ]
        ),
        encoding="utf-8",
    )
    boundary_summary = tmp_path / "ncs_boundary.csv"
    boundary_summary.write_text(
        "\n".join(
            [
                "seed,decoder,loss,lambda_boundary,boundary_method,is_real_feature,status,val_mae,cross_mae,cross_rmse,cross_ssim,cross_gradient_error,cross_edge_mae,run_dir",
                "0,boundary_aux_unet,boundary_aux_l1,0.1,gradient_magnitude,True,SUCCESS,428,420,522,0.823,42,113,seed0",
                "1,boundary_aux_unet,boundary_aux_l1,0.1,gradient_magnitude,True,SUCCESS,422,415,514,0.833,41,112,seed1",
                "2,boundary_aux_unet,boundary_aux_l1,0.1,gradient_magnitude,True,SUCCESS,418,412,510,0.837,42,113,seed2",
            ]
        ),
        encoding="utf-8",
    )
    selected_report = tmp_path / "selected.md"
    selected_report.write_text("selected comparison\nmethods have different training paradigms\nnot benchmark-level proof\n", encoding="utf-8")
    boundary_report = tmp_path / "boundary.md"
    boundary_report.write_text("ncs2d boundary aux decoder probe\nnot benchmark-level proof\n", encoding="utf-8")

    payload = write_protocol_v9_final_selected_report(
        selected_comparison_summary=selected_summary,
        selected_comparison_report=selected_report,
        ncs_boundary_summary=boundary_summary,
        ncs_boundary_report=boundary_report,
        output_dir=tmp_path / "out",
    )
    report_text = payload["report_path"].read_text(encoding="utf-8")
    claims_text = payload["claims_path"].read_text(encoding="utf-8")
    findings_text = payload["key_findings_path"].read_text(encoding="utf-8")
    assert "ncs2d_boundary_aux_decoder" in report_text
    assert "Current best selected candidate" in report_text
    assert "selected comparison" in report_text
    assert "methods have different training paradigms" in report_text
    assert "not benchmark-level proof" in report_text
    assert "NCS improves FWI" not in report_text
    assert "NCS outperforms CNN as benchmark proof" not in report_text
    assert "## Can Claim" in claims_text
    assert "## Cannot Claim" in claims_text
    assert "current best selected candidate, not benchmark winner" in findings_text
