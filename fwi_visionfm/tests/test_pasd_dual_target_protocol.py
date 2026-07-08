import json
from pathlib import Path

from fwi_visionfm.pasd.lock_phase3_assets import lock_assets


def test_phase3_dual_target_protocol_contains_evaluation_only_targets(tmp_path: Path):
    p1 = {
        "source": {"records": "src", "models": "src", "family": "FlatVel-A"},
        "target": {"records": "curve", "models": "curve", "family": "CurveVel-A"},
        "split": {"train": [0], "val": [1], "in_family_test": [2], "cross_family_test": [3, 4]},
        "seed": 0,
    }
    p2 = {
        "target": {"records": "flatfault", "models": "flatfault", "family": "FlatFault-A"},
        "split": {"cross_family_test": [5, 6]},
    }
    decision = {"selected_candidate": "C1_pasd_core_mean"}
    locked = {"lambda_edge": 0.1}
    p1_path = tmp_path / "p1.json"
    p2_path = tmp_path / "p2.json"
    decision_path = tmp_path / "decision.json"
    locked_path = tmp_path / "locked.json"
    for path, payload in ((p1_path, p1), (p2_path, p2), (decision_path, decision), (locked_path, locked)):
        path.write_text(json.dumps(payload), encoding="utf-8")
    config_path, protocol_path = lock_assets(
        p1_path,
        p2_path,
        decision_path,
        locked_path,
        tmp_path / "phase3_config.json",
        tmp_path / "phase3_protocol.json",
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert config["selected_variant"] == "B4_no_geometry_attention"
    assert protocol["targets"]["CurveVel-A"]["role"] == "evaluation_only"
    assert protocol["targets"]["FlatFault-A"]["cross_family_test_indices"] == [5, 6]
