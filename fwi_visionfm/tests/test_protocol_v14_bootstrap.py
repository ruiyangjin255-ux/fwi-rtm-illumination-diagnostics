from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def test_protocol_v14_bootstrap_pairs_sample_ids(tmp_path: Path):
    from scripts.bootstrap_protocol_v14_trace_bridge import bootstrap_protocol_v14_trace_bridge

    root = tmp_path / "protocol_v14"
    root.mkdir()
    candidate = root / "runs" / "flatvel_a_to_curvevel_a" / "dinov2_frozen" / "seed_0" / "B1"
    baseline = root / "runs" / "flatvel_a_to_curvevel_a" / "dinov2_frozen" / "seed_0" / "B0"
    candidate.mkdir(parents=True)
    baseline.mkdir(parents=True)
    candidate_config = {
        "status": "SUCCESS",
        "transfer_id": "flatvel_a_to_curvevel_a",
        "source_family": "flatvel_a",
        "target_family": "curvevel_a",
        "seed": 0,
        "method_id": "M3",
        "method_key": "dinov2_frozen",
        "bridge_id": "B1",
        "bridge_name": "geometry_aware_trace_bridge_geometry",
    }
    baseline_config = {
        **candidate_config,
        "bridge_id": "B0",
        "bridge_name": "raw_envelope_spectrum3",
    }
    (candidate / "config.json").write_text(json.dumps(candidate_config), encoding="utf-8")
    (baseline / "config.json").write_text(json.dumps(baseline_config), encoding="utf-8")
    target = np.zeros((2, 1, 70, 70), dtype=np.float32)
    baseline_pred = np.ones((2, 1, 70, 70), dtype=np.float32)
    candidate_pred = np.full((2, 1, 70, 70), 0.5, dtype=np.float32)
    np.savez_compressed(candidate / "predictions_cross_family_test.npz", velocity_pred_physical=candidate_pred, velocity_true_physical=target, sample_id=np.asarray(["b", "a"], dtype=str))
    np.savez_compressed(baseline / "predictions_cross_family_test.npz", velocity_pred_physical=baseline_pred, velocity_true_physical=target, sample_id=np.asarray(["a", "b"], dtype=str))

    payload = bootstrap_protocol_v14_trace_bridge(root=root, n_bootstrap=50, comparisons=["M3_B1_vs_B0"])
    assert payload["comparison_count"] == 1
    rows = list(csv.DictReader((root / "bootstrap" / "protocol_v14_bootstrap_deltas.csv").open("r", encoding="utf-8")))
    assert rows[0]["sample_ids_aligned"] == "True"
    assert float(rows[0]["delta_mae_mean"]) < 0.0


def test_protocol_v14_bootstrap_accepts_reused_b0_config_without_bridge_id(tmp_path: Path):
    from scripts.bootstrap_protocol_v14_trace_bridge import bootstrap_protocol_v14_trace_bridge

    root = tmp_path / "protocol_v14"
    root.mkdir()
    candidate = root / "runs" / "flatvel_a_to_curvevel_a" / "dinov2_frozen" / "seed_0" / "B1"
    baseline = root / "runs" / "flatvel_a_to_curvevel_a" / "dinov2_frozen" / "seed_0" / "B0"
    candidate.mkdir(parents=True)
    baseline.mkdir(parents=True)
    (candidate / "config.json").write_text(
        json.dumps(
            {
                "status": "SUCCESS",
                "transfer_id": "flatvel_a_to_curvevel_a",
                "source_family": "flatvel_a",
                "target_family": "curvevel_a",
                "seed": 0,
                "method_id": "M3",
                "method_key": "dinov2_frozen",
                "bridge_id": "B1",
                "bridge_name": "geometry_aware_trace_bridge_geometry",
            }
        ),
        encoding="utf-8",
    )
    (baseline / "config.json").write_text(
        json.dumps(
            {
                "status": "SUCCESS",
                "transfer_id": "flatvel_a_to_curvevel_a",
                "source_family": "flatvel_a",
                "target_family": "curvevel_a",
                "seed": 0,
                "method_id": "M3",
                "method_key": "dinov2_frozen",
                "bridge": "raw_envelope_spectrum3",
            }
        ),
        encoding="utf-8",
    )
    target = np.zeros((1, 1, 70, 70), dtype=np.float32)
    np.savez_compressed(candidate / "predictions_cross_family_test.npz", velocity_pred_physical=target, velocity_true_physical=target, sample_id=np.asarray(["a"], dtype=str))
    np.savez_compressed(baseline / "predictions_cross_family_test.npz", velocity_pred_physical=target, velocity_true_physical=target, sample_id=np.asarray(["a"], dtype=str))
    payload = bootstrap_protocol_v14_trace_bridge(root=root, n_bootstrap=20, comparisons=["M3_B1_vs_B0"])
    assert payload["comparison_count"] == 1
