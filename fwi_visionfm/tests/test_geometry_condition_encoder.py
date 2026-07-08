from __future__ import annotations


def test_geometry_condition_encoder_projects_to_token_dim():
    from fwi_visionfm.models.geometry_aware_trace_bridge import GeometryConditionEncoder
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    encoder = GeometryConditionEncoder(condition_dim=10, token_dim=32, hidden_dim=16)
    condition = torch.randn(2, 3, 4, 5, 10)
    output = encoder(condition)
    assert tuple(output.shape) == (2, 3, 4, 5, 32)


def test_geometry_condition_encoder_is_not_rgb_injection():
    from fwi_visionfm.models.geometry_aware_trace_bridge import GeometryConditionEncoder

    encoder = GeometryConditionEncoder(condition_dim=7, token_dim=24, hidden_dim=12)
    assert encoder.condition_dim == 7
    assert encoder.token_dim == 24
