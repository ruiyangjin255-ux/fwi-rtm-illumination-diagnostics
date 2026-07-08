from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image


def test_protocol_v14_report_contains_required_limitations_and_claim_guard(tmp_path: Path):
    from scripts.report_protocol_v14_geometry_aware_trace_bridge import report_protocol_v14_geometry_aware_trace_bridge

    root = tmp_path / "protocol_v14"
    root.mkdir()
    with (root / "protocol_v14_aggregate_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method_id", "bridge_id", "cross_family_MAE", "cross_family_gradient_error"])
        writer.writeheader()
        writer.writerow({"method_id": "M3", "bridge_id": "B0", "cross_family_MAE": 100.0, "cross_family_gradient_error": 10.0})
    with (root / "protocol_v14_geometry_gain.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["comparison_id", "evidence_level"])
        writer.writeheader()
        writer.writerow({"comparison_id": "M3_B1_vs_B0", "evidence_level": "部分或混合证据"})
    with (root / "protocol_v14_per_run_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "status", "missing_required_files"])
        writer.writeheader()
        writer.writerow({"run_id": "r1", "status": "SUCCESS", "missing_required_files": ""})
    with (root / "protocol_v14_unsuccessful_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "status", "skip_reason"])
        writer.writeheader()
    with (root / "protocol_v14_incomplete_outputs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_id", "status", "missing_required_files"])
        writer.writeheader()
        writer.writerow({"run_id": "r0", "status": "SUCCESS", "missing_required_files": "geometry_metadata.json"})
    (root / "protocol_v14_summary.json").write_text('{"run_count":1,"success":1,"unsuccessful_count":0,"incomplete_output_count":1}', encoding="utf-8")
    (root / "geometry_audit.json").write_text('{"geometry_provenance":"CANONICAL_RECONSTRUCTED"}', encoding="utf-8")
    bootstrap_dir = root / "bootstrap"
    bootstrap_dir.mkdir()
    with (bootstrap_dir / "protocol_v14_bootstrap_deltas.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["comparison_id", "seed", "delta_mae_mean", "mae_ci_low", "mae_ci_high"])
        writer.writeheader()
        writer.writerow({"comparison_id": "M3_B1_vs_B0", "seed": 0, "delta_mae_mean": -1.0, "mae_ci_low": -1.5, "mae_ci_high": -0.5})
    run_dir = root / "runs" / "flatvel_a_to_curvevel_a" / "dinov2_frozen" / "seed_0" / "B0"
    run_dir.mkdir(parents=True)
    Image.fromarray(np.full((8, 8, 3), 120, dtype=np.uint8), mode="RGB").save(run_dir / "prediction_grid.png")
    np.savez_compressed(
        run_dir / "predictions_cross_family_test.npz",
        velocity_pred_physical=np.zeros((2, 1, 70, 70), dtype=np.float32),
        velocity_true_physical=np.ones((2, 1, 70, 70), dtype=np.float32),
        sample_id=np.asarray(["a", "b"], dtype=str),
    )
    b3_dir = root / "runs" / "flatvel_a_to_curvevel_a" / "dinov2_frozen" / "seed_0" / "B3"
    b3_dir.mkdir(parents=True)
    np.savez_compressed(
        b3_dir / "predictions_cross_family_test.npz",
        velocity_pred_physical=np.full((2, 1, 70, 70), 0.2, dtype=np.float32),
        velocity_true_physical=np.ones((2, 1, 70, 70), dtype=np.float32),
        sample_id=np.asarray(["a", "b"], dtype=str),
    )
    robustness_dir = root / "robustness"
    robustness_dir.mkdir()
    with (robustness_dir / "protocol_v14_robustness_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["transfer_id", "seed", "method_id", "bridge_id", "perturbation", "metric_name", "metric_value", "degradation", "status", "note"])
        writer.writeheader()
        writer.writerow({"transfer_id": "flatvel_a_to_curvevel_a", "seed": 0, "method_id": "M3", "bridge_id": "B0", "perturbation": "clean", "metric_name": "mae", "metric_value": 1.0, "degradation": 0.0, "status": "AVAILABLE_CLEAN_ONLY", "note": ""})
        writer.writerow({"transfer_id": "flatvel_a_to_curvevel_a", "seed": 0, "method_id": "M3", "bridge_id": "B3", "perturbation": "few_shot_3", "metric_name": "mae", "metric_value": "", "degradation": "", "status": "UNAVAILABLE_NO_CHECKPOINT", "note": ""})
    payload = report_protocol_v14_geometry_aware_trace_bridge(root=root)
    report_text = Path(payload["report_path"]).read_text(encoding="utf-8")
    assert "不构成标准基准级结论" in report_text
    assert "不能声称 geometry-aware trace bridge 已证明提升 FWI 泛化能力" in report_text
    assert "CANONICAL_RECONSTRUCTED" in report_text
    assert "输出不完整 run 数" in report_text
    assert "geometry_metadata.json" in report_text
    assert (root / "figures" / "figure_01_geometry_provenance.png").is_file()
    assert (root / "figures" / "figure_08_bootstrap_trace_bridge_effect.png").is_file()
    assert (root / "figures" / "figure_09_robustness_degradation.png").is_file()
    assert (root / "figures" / "figure_10_prediction_grid_bridge_and_robustness.png").is_file()
