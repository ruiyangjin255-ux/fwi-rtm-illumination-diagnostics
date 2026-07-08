from __future__ import annotations

import json
from pathlib import Path


def _write_run(path: Path, *, seed: int, mae: float, rmse: float, ssim: float, gradient_error: float, edge_mae: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "decoder_name": "boundary_aux_unet",
                "loss_name": "boundary_aux_l1",
                "lambda_boundary": 0.10,
                "boundary_method": "gradient_magnitude",
                "is_real_feature": True,
                "status": "SUCCESS",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (path / "metrics_cross_family_test.json").write_text(
        json.dumps(
            {"mae": mae, "rmse": rmse, "ssim": ssim, "gradient_error": gradient_error, "edge_mae": edge_mae, "metric_space": "physical_velocity"},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (path / "metrics_val.json").write_text(json.dumps({"mae": mae - 1, "rmse": rmse - 1, "ssim": ssim, "metric_space": "physical_velocity"}, indent=2), encoding="utf-8")


def test_protocol_v9_ncs2d_boundary_aux_decoder_report_contains_required_language(tmp_path: Path):
    from fwi_visionfm.scripts.report_protocol_v9_ncs2d_boundary_aux_decoder_probe import write_protocol_v9_ncs2d_boundary_aux_decoder_probe_report

    root = tmp_path / "probe"
    _write_run(root / "seed_0", seed=0, mae=420, rmse=520, ssim=0.83, gradient_error=50, edge_mae=120)
    _write_run(root / "seed_1", seed=1, mae=419, rmse=519, ssim=0.82, gradient_error=51, edge_mae=121)
    _write_run(root / "seed_2", seed=2, mae=418, rmse=518, ssim=0.81, gradient_error=52, edge_mae=122)

    selected_root = tmp_path / "selected"
    selected_root.mkdir(parents=True, exist_ok=True)
    (selected_root / "protocol_v9_selected_comparison_report.md").write_text("selected comparison stub\nnot benchmark-level proof\n", encoding="utf-8")
    ncs_root = tmp_path / "ncs"
    ncs_root.mkdir(parents=True, exist_ok=True)
    (ncs_root / "protocol_v9_ncs2d_seed_stability_report.md").write_text("ncs2d seed stability stub\n", encoding="utf-8")
    v7_root = tmp_path / "v7"
    v7_root.mkdir(parents=True, exist_ok=True)
    (v7_root / "protocol_v7_boundary_auxiliary_seed_stability_summary.csv").write_text("seed,baseline_MAE,boundary_MAE\n0,1,1\n", encoding="utf-8")

    payload = write_protocol_v9_ncs2d_boundary_aux_decoder_probe_report(
        root=root,
        selected_comparison_root=selected_root,
        ncs2d_root=ncs_root,
        v7_boundary_root=v7_root,
        output_dir=root,
    )
    report_text = payload["report_path"].read_text(encoding="utf-8")
    assert "ncs2d boundary aux decoder" in report_text.lower()
    assert "not benchmark-level proof" in report_text
    assert "gradient_magnitude" in report_text
