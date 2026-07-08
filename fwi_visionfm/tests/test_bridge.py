import torch

from fwi_visionfm.pasd.bridge import HybridAttributeBridge


def test_hybrid_bridge_shapes_and_finiteness():
    bridge = HybridAttributeBridge(lowpass_kernel=9)
    records = torch.randn(2, 5, 64, 16)
    out = bridge(records)
    assert out.attributes.shape == (2, 5, 3, 64, 16)
    assert out.geometry.shape == (2, 5, 3)
    assert torch.isfinite(out.attributes).all()
    assert torch.isfinite(out.geometry).all()
