from __future__ import annotations

import csv
import json
from pathlib import Path


def test_protocol_v14_summary_uses_actual_run_status_and_writes_unsuccessful_outputs(tmp_path: Path) -> None:
    from scripts.summarize_protocol_v14_geometry_aware_trace_bridge import summarize_protocol_v14_geometry_aware_trace_bridge

    root = tmp_path / "protocol_v14"
    root.mkdir()
    matrix_path = root / "protocol_v14_run_matrix.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "transfer_id",
                "source_family",
                "target_family",
                "method_id",
                "method_key",
                "method_name",
                "bridge_id",
                "bridge_name",
                "seed",
                "status",
                "reused_from",
                "metric_space",
                "skip_reason",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": "r1",
                "transfer_id": "flatvel_a_to_curvevel_a",
                "source_family": "flatvel_a",
                "target_family": "curvevel_a",
                "method_id": "M3",
                "method_key": "dinov2_frozen",
                "method_name": "DINOv2 frozen",
                "bridge_id": "B1",
                "bridge_name": "geometry_aware_trace_bridge_geometry",
                "seed": "0",
                "status": "PENDING",
                "reused_from": "",
                "metric_space": "physical_velocity",
                "skip_reason": "",
            }
        )
        writer.writerow(
            {
                "run_id": "r2",
                "transfer_id": "flatvel_a_to_curvevel_a",
                "source_family": "flatvel_a",
                "target_family": "curvevel_a",
                "method_id": "M6",
                "method_key": "ncs2d_frozen",
                "method_name": "NCS2D frozen",
                "bridge_id": "B0",
                "bridge_name": "raw_envelope_spectrum3",
                "seed": "0",
                "status": "REUSE_GATE_PASSED",
                "reused_from": "",
                "metric_space": "physical_velocity",
                "skip_reason": "",
            }
        )

    success_dir = root / "runs" / "flatvel_a_to_curvevel_a" / "dinov2_frozen" / "seed_0" / "B1"
    success_dir.mkdir(parents=True)
    success_config = {"status": "SUCCESS", "metric_space": "physical_velocity", "reused_from": "", "is_real_feature": False}
    (success_dir / "config.json").write_text(json.dumps(success_config), encoding="utf-8")
    for name in [
        "config_hash.txt",
        "model_card.json",
        "geometry_metadata.json",
        "train_history.csv",
        "metrics_val.json",
        "metrics_in_family_test.json",
        "metrics_cross_family_test.json",
        "predictions_in_family_test.npz",
        "predictions_cross_family_test.npz",
        "prediction_grid.png",
        "gradient_grid.png",
        "run_log.txt",
    ]:
        path = success_dir / name
        if name.endswith(".json"):
            payload = {"metric_space": "physical_velocity", "mae": 1.0, "rmse": 2.0, "ssim": 0.5, "gradient_error": 3.0, "edge_mae": 4.0}
            if name == "model_card.json":
                payload = {"x": 1}
            if name == "geometry_metadata.json":
                payload = {"geometry_provenance": "CANONICAL_RECONSTRUCTED"}
            path.write_text(json.dumps(payload), encoding="utf-8")
        elif name.endswith(".csv"):
            path.write_text("epoch,train_l1\n1,1.0\n", encoding="utf-8")
        else:
            path.write_bytes(b"x")

    failed_dir = root / "runs" / "flatvel_a_to_curvevel_a" / "ncs2d_frozen" / "seed_0" / "B0"
    failed_dir.mkdir(parents=True)
    failed_config = {"status": "FAILED", "metric_space": "physical_velocity", "skip_reason": "example failure"}
    (failed_dir / "config.json").write_text(json.dumps(failed_config), encoding="utf-8")

    summary = summarize_protocol_v14_geometry_aware_trace_bridge(root=root)
    assert summary["run_count"] == 2
    assert summary["success"] == 1
    assert summary["failed"] == 1
    assert summary["unsuccessful_count"] == 1

    unsuccessful = (root / "protocol_v14_unsuccessful_runs.csv").read_text(encoding="utf-8")
    assert "r2" in unsuccessful
    per_run = (root / "protocol_v14_per_run_metrics.csv").read_text(encoding="utf-8")
    assert "SUCCESS" in per_run
    assert "FAILED" in per_run
