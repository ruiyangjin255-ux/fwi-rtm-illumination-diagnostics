from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "bridge_name",
    [
        "raw_repeat3",
        "raw_normalized3",
        "envelope_repeat3",
        "raw_plus_envelope",
        "raw_spectrogram",
        "spectrogram_multiband",
        "raw_envelope_spectrum3",
        "low_high_frequency_bridge",
    ],
)
def test_bridge_registry_handles_fake_records_without_nan(bridge_name: str):
    torch = pytest.importorskip("torch")
    from fwi_visionfm.data.bridge_registry import build_bridge

    records = torch.randn(2, 3, 32, 24)
    bridge = build_bridge(bridge_name, {"output_size": [32, 32]})
    out = bridge.forward(records)
    image = out["image"]

    assert image.shape == (2, 3, 32, 32)
    assert out["bridge_name"] == bridge_name
    assert out["metadata"]["input_shape"] == [2, 3, 32, 24]
    assert out["metadata"]["output_shape"] == [2, 3, 32, 32]
    assert torch.isfinite(image).all()


def test_bridge_registry_handles_single_unbatched_record():
    torch = pytest.importorskip("torch")
    from fwi_visionfm.data.bridge_registry import build_bridge

    records = torch.randn(3, 32, 24)
    out = build_bridge("raw_plus_envelope", {"output_size": [16, 16]}).forward(records)

    assert out["image"].shape == (3, 16, 16)
