from __future__ import annotations

import csv
import json
from pathlib import Path


def _write_run(root: Path, *, run_id: str, seed: int, model_type: str, decoder: str, loss: str, status: str, skip_reason: str, val: dict, cross: dict, lambda_boundary: str = "", boundary_method: str = "") -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "seed": seed,
                "decoder": decoder,
                "loss": loss,
                "lambda_boundary": None if lambda_boundary == "" else float(lambda_boundary),
                "boundary_method": boundary_method or None,
                "status": status,
                "skip_reason": skip_reason,
                "model_type": model_type,
            }
        ),
        encoding="utf-8",
    )
    if val:
        (run_dir / "metrics_val.json").write_text(json.dumps(val), encoding="utf-8")
    if cross:
        (run_dir / "metrics_cross_family_test.json").write_text(json.dumps(cross), encoding="utf-8")


def test_seed_stability_summary_merges_reuse_seed0_and_counts_wins(tmp_path: Path):
    from fwi_visionfm.scripts.run_protocol_v7_boundary_auxiliary_seed_stability import (
        build_seed_stability_summary,
        compute_seed_stability_win_counts,
    )

    root = tmp_path / "seed_stability"
    reuse_root = tmp_path / "reuse_seed0"

    _write_run(
        reuse_root,
        run_id="run_1",
        seed=0,
        model_type="baseline",
        decoder="unet_decoder",
        loss="default_l1",
        status="SUCCESS",
        skip_reason="",
        val={"mae": 1.0, "rmse": 2.0, "ssim": 0.80},
        cross={"mae": 1.1, "rmse": 2.1, "ssim": 0.70, "gradient_error": 0.30, "edge_mae": 0.40},
    )
    _write_run(
        reuse_root,
        run_id="run_3",
        seed=0,
        model_type="boundary_aux",
        decoder="boundary_aux_unet",
        loss="boundary_aux_l1",
        status="SUCCESS",
        skip_reason="",
        lambda_boundary="0.1",
        boundary_method="gradient_magnitude",
        val={"mae": 0.9, "rmse": 1.9, "ssim": 0.81, "boundary_val_l1": 0.2},
        cross={"mae": 1.0, "rmse": 2.0, "ssim": 0.71, "gradient_error": 0.28, "edge_mae": 0.39},
    )
    _write_run(
        root,
        run_id="seed_1_baseline",
        seed=1,
        model_type="baseline",
        decoder="unet_decoder",
        loss="default_l1",
        status="SUCCESS",
        skip_reason="",
        val={"mae": 1.2, "rmse": 2.2, "ssim": 0.79},
        cross={"mae": 1.3, "rmse": 2.3, "ssim": 0.69, "gradient_error": 0.33, "edge_mae": 0.43},
    )
    _write_run(
        root,
        run_id="seed_1_boundary_aux",
        seed=1,
        model_type="boundary_aux",
        decoder="boundary_aux_unet",
        loss="boundary_aux_l1",
        status="SUCCESS",
        skip_reason="",
        lambda_boundary="0.1",
        boundary_method="gradient_magnitude",
        val={"mae": 1.1, "rmse": 2.1, "ssim": 0.80, "boundary_val_l1": 0.25},
        cross={"mae": 1.2, "rmse": 2.2, "ssim": 0.70, "gradient_error": 0.31, "edge_mae": 0.42},
    )
    _write_run(
        root,
        run_id="seed_2_baseline",
        seed=2,
        model_type="baseline",
        decoder="unet_decoder",
        loss="default_l1",
        status="FAILED",
        skip_reason="oom",
        val={},
        cross={},
    )

    summary_path = build_seed_stability_summary(root=root, reuse_seed0_root=reuse_root)
    rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8")))
    assert len(rows) == 6
    assert {row["seed"] for row in rows} == {"0", "1", "2"}
    assert any(row["model_type"] == "boundary_aux" and row["seed"] == "0" for row in rows)
    assert any(row["status"] == "FAILED" and row["skip_reason"] == "oom" for row in rows)
    assert any(row["status"] == "SKIPPED" and row["seed"] == "2" and row["model_type"] == "boundary_aux" for row in rows)

    wins = compute_seed_stability_win_counts(rows)
    assert wins["gradient_error_lower"] == 2
    assert wins["edge_MAE_lower"] == 2
    assert wins["MAE_lower"] == 2
    assert wins["RMSE_lower"] == 2
    assert wins["SSIM_higher"] == 2
