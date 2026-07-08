from __future__ import annotations

import json
from pathlib import Path


def test_geometry_audit_detects_canonical_reconstructed_when_no_real_metadata(tmp_path: Path):
    from scripts.audit_protocol_v14_geometry_metadata import audit_protocol_v14_geometry_metadata

    repo_root = tmp_path / "repo"
    data_root = repo_root / "data"
    data_root.mkdir(parents=True)
    config_path = repo_root / "config.yaml"
    config_path.write_text("protocol: protocol_v14\nfamilies: [flatvel_a, curvevel_a, flatfault_a]\n", encoding="utf-8")
    output_dir = tmp_path / "out"
    payload = audit_protocol_v14_geometry_metadata(repo_root=repo_root, config_path=config_path, output_dir=output_dir)
    assert payload["geometry_provenance"] == "CANONICAL_RECONSTRUCTED"
    assert (output_dir / "geometry_audit.json").is_file()
    saved = json.loads((output_dir / "geometry_audit.json").read_text(encoding="utf-8"))
    assert saved["geometry_provenance"] == "CANONICAL_RECONSTRUCTED"


def test_geometry_audit_detects_real_metadata_fields(tmp_path: Path):
    from scripts.audit_protocol_v14_geometry_metadata import determine_geometry_provenance

    payload = determine_geometry_provenance(
        [
            {"field_name": "source_x", "available": True, "source": "manifest"},
            {"field_name": "receiver_x", "available": True, "source": "manifest"},
            {"field_name": "dt", "available": True, "source": "config"},
        ]
    )
    assert payload == "REAL_METADATA"
