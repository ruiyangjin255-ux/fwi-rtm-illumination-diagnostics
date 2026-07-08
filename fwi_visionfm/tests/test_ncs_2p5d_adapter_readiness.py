from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def test_ncs_2p5d_pending_writes_status_report(tmp_path: Path):
    from fwi_visionfm.models.seismic_backbones.ncs_backbone import inspect_ncs_2p5d_adapter

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "config.json").write_text(json.dumps({"model_type": "vit25d", "image_size": 224, "patch_size": 16}), encoding="utf-8")
    (weights_dir / "model.safetensors").write_bytes(b"fake")

    report_path = tmp_path / "adapter_status_report.json"
    payload = inspect_ncs_2p5d_adapter(repo_root=repo_root, weights_path=weights_dir, status_report_path=report_path, device="cpu")

    assert payload["status"] == "WEIGHTS_PRESENT_ADAPTER_PENDING"
    assert payload["is_real_feature"] is False
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "pending_reason" in report
    assert report["status"] == "WEIGHTS_PRESENT_ADAPTER_PENDING"


def test_ncs_2p5d_mock_builder_can_forward(tmp_path: Path, monkeypatch):
    from fwi_visionfm.models.seismic_backbones import ncs_backbone

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "config.json").write_text(json.dumps({"model_type": "vit25d", "image_size": 224, "patch_size": 16}), encoding="utf-8")
    (weights_dir / "model.safetensors").write_bytes(b"fake")

    class FakeModel:
        def encode(self, inputs):
            array = np.asarray(inputs, dtype=np.float32)
            return array.mean(axis=(2, 3, 4))

    def _fake_loader(*args, **kwargs):
        return FakeModel(), {
            "builder_name": "mock_builder",
            "backend": "repo_builder",
            "pseudo_2p5d_from_shot_gather": True,
        }

    monkeypatch.setattr(ncs_backbone, "_attempt_load_ncs_2p5d_repo_model", _fake_loader)

    payload = ncs_backbone.inspect_ncs_2p5d_adapter(repo_root=repo_root, weights_path=weights_dir, status_report_path=tmp_path / "ok.json", device="cpu")

    assert payload["status"] == "AVAILABLE"
    assert payload["is_real_feature"] is True
    features = payload["model"].encode(np.random.randn(2, 4, 3, 224, 224).astype(np.float32))
    assert np.asarray(features).shape == (2, 4)
