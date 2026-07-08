from __future__ import annotations


def test_physics_aware_token_fusion_preserves_token_shape():
    from fwi_visionfm.models.geometry_aware_trace_bridge import PhysicsAwareTokenFusion
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    fusion = PhysicsAwareTokenFusion(token_dim=16, condition_dim=16)
    tokens = torch.randn(2, 3, 4, 5, 16)
    condition = torch.randn(2, 3, 4, 5, 16)
    output = fusion(tokens, condition)
    assert tuple(output.shape) == tuple(tokens.shape)


def test_geometry_aware_trace_bridge_combines_geometry_context_and_tokens():
    from fwi_visionfm.models.geometry_aware_trace_bridge import GeometryAwareTraceBridge
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    bridge = GeometryAwareTraceBridge(token_dim=8, use_multiscale_context=True)
    patch_tokens = torch.randn(1, 2, 3, 4, 8)
    records = torch.randn(1, 2, 3, 16)
    g_trace = torch.randn(1, 2, 3, 4, 7)
    g_time = torch.randn(1, 2, 3, 4, 3)
    multiscale = torch.randn(1, 2, 3, 4, 3)
    out = bridge.forward(
        patch_tokens=patch_tokens,
        records=records,
        g_trace=g_trace,
        g_time=g_time,
        multiscale_feature=multiscale,
    )
    assert tuple(out["tokens"].shape) == (1, 2, 3, 4, 8)
    assert out["metadata"]["trace_context_radius"] == 2
