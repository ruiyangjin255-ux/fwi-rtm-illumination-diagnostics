from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def _write_fake_hf_vit(path: Path) -> None:
    import torch
    from transformers import ViTConfig, ViTModel

    cfg = ViTConfig(
        image_size=224,
        patch_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_channels=3,
    )
    model = ViTModel(cfg, add_pooling_layer=False)
    model.save_pretrained(str(path))
    torch.save(model.state_dict(), path / "weights.pt")


def test_ncs_2d_adapter_transformers_forward_and_resize(tmp_path: Path):
    from fwi_visionfm.models.seismic_backbones.ncs_backbone import NCS2DBackboneAdapter

    weights_dir = tmp_path / "ncs_2d_fake"
    weights_dir.mkdir(parents=True, exist_ok=True)
    _write_fake_hf_vit(weights_dir)

    adapter = NCS2DBackboneAdapter.from_pretrained(weights_dir, feature_mode="mean_patch", device="cpu")
    batch = np.random.randn(2, 3, 64, 64).astype(np.float32)
    features = adapter.encode(batch)

    assert features.shape == (2, 32)
    assert adapter.metadata["model_name"] == "ncs_2d"
    assert adapter.metadata["load_backend"] == "transformers"
    assert adapter.metadata["feature_mode"] == "mean_patch"
    assert adapter.metadata["is_real_feature"] is True
    assert adapter.metadata["input_size"] == 224
    assert adapter.metadata["hidden_size"] == 32


def test_ncs_2d_adapter_supports_cls_mode(tmp_path: Path):
    from fwi_visionfm.models.seismic_backbones.ncs_backbone import NCS2DBackboneAdapter

    weights_dir = tmp_path / "ncs_2d_fake"
    weights_dir.mkdir(parents=True, exist_ok=True)
    _write_fake_hf_vit(weights_dir)

    adapter = NCS2DBackboneAdapter.from_pretrained(weights_dir, feature_mode="cls", device="cpu")
    features = adapter.encode(np.random.randn(1, 3, 224, 224).astype(np.float32))

    assert features.shape == (1, 32)
    assert adapter.metadata["feature_mode"] == "cls"


def test_ncs_2d_adapter_can_return_patch_tokens(tmp_path: Path):
    from fwi_visionfm.models.seismic_backbones.ncs_backbone import NCS2DBackboneAdapter

    weights_dir = tmp_path / "ncs_2d_fake"
    weights_dir.mkdir(parents=True, exist_ok=True)
    _write_fake_hf_vit(weights_dir)

    adapter = NCS2DBackboneAdapter.from_pretrained(weights_dir, feature_mode="mean_patch", device="cpu")
    tokens = adapter.encode_tokens(np.random.randn(2, 3, 224, 224).astype(np.float32))

    assert tokens.ndim == 3
    assert tokens.shape[0] == 2
    assert tokens.shape[-1] == 32


def test_ncs_2d_adapter_load_failure_is_clear(tmp_path: Path):
    from fwi_visionfm.models.seismic_backbones.ncs_backbone import NCS2DBackboneAdapter

    broken = tmp_path / "broken"
    broken.mkdir(parents=True, exist_ok=True)
    (broken / "config.json").write_text(json.dumps({"model_type": "vit"}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Failed to load NCS 2D transformers adapter"):
        NCS2DBackboneAdapter.from_pretrained(broken, feature_mode="mean_patch", device="cpu")
