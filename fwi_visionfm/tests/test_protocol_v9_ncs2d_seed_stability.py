from __future__ import annotations

import json
from pathlib import Path


def _write_probe_run(path: Path, *, seed: int, mae: float, rmse: float, ssim: float, gradient_error: float, edge_mae: float, reused_from: str = "") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "feature_cache": "outputs/protocol_v9_ncs_adapter_repair/feature_cache/ncs_2d",
                "decoder_name": "lightweight_feature_decoder",
                "loss_name": "default_l1",
                "epochs": 2,
                "status": "SUCCESS",
                "is_real_feature": True,
                "feature_cache_status": "AVAILABLE",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (path / "metrics_val.json").write_text(json.dumps({"mae": mae - 1.0, "rmse": rmse - 1.0, "ssim": ssim, "metric_space": "physical_velocity"}, indent=2), encoding="utf-8")
    (path / "metrics_cross_family_test.json").write_text(
        json.dumps(
            {"mae": mae, "rmse": rmse, "ssim": ssim, "gradient_error": gradient_error, "edge_mae": edge_mae, "metric_space": "physical_velocity"},
            indent=2,
        ),
        encoding="utf-8",
    )
    (path / "run_log.txt").write_text(f"reused_from={reused_from}\n", encoding="utf-8")


def test_protocol_v9_ncs2d_seed_stability_summary_handles_reuse_and_new_runs(tmp_path: Path):
    from fwi_visionfm.scripts.report_protocol_v9_ncs2d_seed_stability import collect_seed_rows, write_protocol_v9_ncs2d_seed_stability_report

    root = tmp_path / "protocol_v9_ncs2d_seed_stability"
    (root / "decoder_probe").mkdir(parents=True, exist_ok=True)
    seed0_dir = tmp_path / "seed0"
    _write_probe_run(seed0_dir, seed=0, mae=423.7, rmse=523.9, ssim=0.83, gradient_error=55.6, edge_mae=135.1, reused_from="adapter_repair_seed0")
    _write_probe_run(root / "decoder_probe" / "ncs_2d_seed1", seed=1, mae=430.0, rmse=530.0, ssim=0.82, gradient_error=57.0, edge_mae=138.0)
    _write_probe_run(root / "decoder_probe" / "ncs_2d_seed2", seed=2, mae=428.0, rmse=528.0, ssim=0.81, gradient_error=56.0, edge_mae=136.0)

    rows = collect_seed_rows(root=root, seed0_dir=seed0_dir)
    assert len(rows) == 3
    assert rows[0]["reused_from"] != ""
    assert rows[1]["reused_from"] == ""
    assert all(row["is_real_feature"] is True for row in rows)

    adapter_report = tmp_path / "adapter_report.md"
    adapter_report.write_text("previous ncs_2d IMPORT_ERROR repaired\nncs_2p5d remains adapter pending\nnot benchmark-level proof\n", encoding="utf-8")
    payload = write_protocol_v9_ncs2d_seed_stability_report(root=root, seed0_dir=seed0_dir, adapter_repair_report=adapter_report, output_dir=root)
    assert payload["summary_path"].exists()
