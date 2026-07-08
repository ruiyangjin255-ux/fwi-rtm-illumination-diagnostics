from __future__ import annotations


def test_geometry_aware_bridge_default_does_not_change_old_bridge():
    from fwi_visionfm.data.geometry_aware_bridge import GeometryAwareBridgeWrapper
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    records = torch.randn(2, 3, 32, 24)
    bridge = GeometryAwareBridgeWrapper("raw_envelope_spectrum3", {"output_size": [32, 32], "geometry": {"enabled": False}})
    out = bridge.forward(records)
    assert tuple(out["image"].shape) == (2, 3, 32, 32)
    assert "geometry_config" not in out["metadata"]


def test_geometry_aware_bridge_concat_and_projection_work():
    from fwi_visionfm.data.geometry_aware_bridge import GeometryAwareBridgeWrapper
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    records = torch.randn(2, 3, 32, 24)
    bridge = GeometryAwareBridgeWrapper(
        "raw_envelope_spectrum3",
        {
            "output_size": [32, 32],
            "geometry": {"enabled": True, "mode": "sinusoidal", "fusion": "concat", "projection_to_3ch": True},
        },
    )
    out = bridge.forward(records)
    assert tuple(out["image"].shape) == (2, 3, 32, 32)
    assert out["metadata"]["geometry_config"]["enabled"] is True

