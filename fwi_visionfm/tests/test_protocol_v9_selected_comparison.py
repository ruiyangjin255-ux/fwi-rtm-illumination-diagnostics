from __future__ import annotations

import json
from pathlib import Path


def _write_run(path: Path, *, seed: int, method: str, backbone: str, decoder: str, loss: str, is_real_feature: bool, mae: float, rmse: float, ssim: float, gradient_error: float, edge_mae: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "backbone": backbone,
                "decoder_name": decoder,
                "loss_name": loss,
                "epochs": 2,
                "status": "SUCCESS",
                "is_real_feature": is_real_feature,
                "feature_cache_status": "AVAILABLE" if is_real_feature else "",
                "method_name": method,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (path / "metrics_cross_family_test.json").write_text(
        json.dumps(
            {
                "mae": mae,
                "rmse": rmse,
                "ssim": ssim,
                "gradient_error": gradient_error,
                "edge_mae": edge_mae,
                "metric_space": "physical_velocity",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_protocol_v9_selected_comparison_collects_four_methods(tmp_path: Path):
    from fwi_visionfm.scripts.report_protocol_v9_selected_comparison import collect_selected_rows

    ncs_root = tmp_path / "ncs"
    vit_seed_root = tmp_path / "vit_seed"
    vit_seed0 = tmp_path / "vit_seed0"
    _write_run(vit_seed0, seed=0, method="vit", backbone="vit_mae_base", decoder="lightweight_feature_decoder", loss="default_l1", is_real_feature=True, mae=420, rmse=520, ssim=0.82, gradient_error=60, edge_mae=130)
    _write_run(vit_seed_root / "vit_mae_base_seed1", seed=1, method="vit", backbone="vit_mae_base", decoder="lightweight_feature_decoder", loss="default_l1", is_real_feature=True, mae=421, rmse=521, ssim=0.81, gradient_error=61, edge_mae=131)
    _write_run(vit_seed_root / "vit_mae_base_seed2", seed=2, method="vit", backbone="vit_mae_base", decoder="lightweight_feature_decoder", loss="default_l1", is_real_feature=True, mae=422, rmse=522, ssim=0.80, gradient_error=62, edge_mae=132)
    _write_run(ncs_root / "decoder_probe" / "ncs_2d_seed1", seed=1, method="ncs", backbone="ncs_2d", decoder="lightweight_feature_decoder", loss="default_l1", is_real_feature=True, mae=418, rmse=518, ssim=0.83, gradient_error=59, edge_mae=129)
    _write_run(ncs_root / "decoder_probe" / "ncs_2d_seed2", seed=2, method="ncs", backbone="ncs_2d", decoder="lightweight_feature_decoder", loss="default_l1", is_real_feature=True, mae=417, rmse=517, ssim=0.82, gradient_error=60, edge_mae=128)
    _write_run(tmp_path / "ncs_seed0", seed=0, method="ncs", backbone="ncs_2d", decoder="lightweight_feature_decoder", loss="default_l1", is_real_feature=True, mae=419, rmse=519, ssim=0.81, gradient_error=58, edge_mae=127)

    v7_summary = tmp_path / "protocol_v7_selected_multiseed_summary.csv"
    v7_summary.write_text(
        "\n".join(
            [
                "seed,baseline_MAE,boundary_MAE,baseline_RMSE,boundary_RMSE,baseline_SSIM,boundary_SSIM,baseline_gradient_error,boundary_gradient_error,baseline_edge_MAE,boundary_edge_MAE,baseline_status,boundary_status",
                "0,430,429,525,521,0.81,0.80,41,40,112,111,SUCCESS,SUCCESS",
                "1,414,421,519,525,0.84,0.84,46,44,123,119,SUCCESS,SUCCESS",
                "2,469,463,597,581,0.78,0.79,47,45,125,122,SUCCESS,SUCCESS",
            ]
        ),
        encoding="utf-8",
    )

    rows = collect_selected_rows(
        ncs2d_root=ncs_root,
        ncs2d_seed0_dir=tmp_path / "ncs_seed0",
        vit_mae_seed0_dir=vit_seed0,
        vit_mae_seed_root=vit_seed_root,
        v7_boundary_root=tmp_path / "unused",
        v7_selected_root=tmp_path,
    )
    method_names = {row["method_name"] for row in rows}
    assert method_names == {
        "cnn_baseline_unet_l1",
        "boundary_aux_gradient_lambda010",
        "vit_mae_base_frozen_decoder",
        "ncs2d_frozen_decoder",
    }
