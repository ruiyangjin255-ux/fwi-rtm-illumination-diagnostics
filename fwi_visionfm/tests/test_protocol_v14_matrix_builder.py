from __future__ import annotations

import json
from pathlib import Path


def test_protocol_v14_matrix_builder_creates_72_runs_and_b0_reuse_flags(tmp_path: Path):
    from scripts.build_protocol_v14_matrix import build_protocol_v14_matrix

    config = tmp_path / "protocol_v14.yaml"
    config.write_text(
        "\n".join(
            [
                "protocol: protocol_v14_geometry_aware_trace_bridge",
                "seeds: [0, 1, 2]",
                "shot_count: 5",
                "image_size: 224",
                "epochs: 2",
                "decoder: common_bounded_velocity_decoder",
                "loss: default_l1",
                "metric_space: physical_velocity",
            ]
        ),
        encoding="utf-8",
    )
    geometry_audit = tmp_path / "geometry_audit.json"
    geometry_audit.write_text(json.dumps({"geometry_provenance": "CANONICAL_RECONSTRUCTED"}), encoding="utf-8")
    reuse_gate = tmp_path / "reuse.json"
    reuse_gate.write_text(json.dumps({"rows": [{"run_id": "flatvel_a_to_curvevel_a__dinov2_frozen__B0__seed0", "reusable": True, "source_run_dir": "x"}]}), encoding="utf-8")
    out = tmp_path / "protocol_v14"
    payload = build_protocol_v14_matrix(repo_root=tmp_path, config_path=config, geometry_audit_path=geometry_audit, reuse_gate_path=reuse_gate, output_dir=out)
    assert payload["run_count"] == 72
    assert any(row["bridge_id"] == "B0" for row in payload["rows"])
    assert (out / "protocol_v14_run_matrix.csv").is_file()


def test_protocol_v14_matrix_builder_skips_when_geometry_unavailable(tmp_path: Path):
    from scripts.build_protocol_v14_matrix import build_protocol_v14_matrix

    config = tmp_path / "protocol_v14.yaml"
    config.write_text("protocol: protocol_v14\nseeds: [0]\nshot_count: 5\nimage_size: 224\nepochs: 2\ndecoder: common_bounded_velocity_decoder\nloss: default_l1\nmetric_space: physical_velocity\n", encoding="utf-8")
    geometry_audit = tmp_path / "geometry_audit.json"
    geometry_audit.write_text(json.dumps({"geometry_provenance": "UNAVAILABLE"}), encoding="utf-8")
    reuse_gate = tmp_path / "reuse.json"
    reuse_gate.write_text(json.dumps({"rows": []}), encoding="utf-8")
    payload = build_protocol_v14_matrix(repo_root=tmp_path, config_path=config, geometry_audit_path=geometry_audit, reuse_gate_path=reuse_gate, output_dir=tmp_path / "out")
    assert all(row["status"] == "SKIPPED_GEOMETRY_UNAVAILABLE" for row in payload["rows"])
