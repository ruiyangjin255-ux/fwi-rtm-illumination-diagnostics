from __future__ import annotations


def test_shot_aggregators_shapes_and_attention_weights():
    from fwi_visionfm.models.aggregators import AttentionShotAggregator, MeanShotAggregator, SourceAwareAttentionAggregator
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    x = torch.randn(2, 4, 16)
    mean_out = MeanShotAggregator()(x)
    assert tuple(mean_out["aggregated_feature"].shape) == (2, 16)
    assert tuple(mean_out["attention_weights"].shape) == (2, 4)

    attn_out = AttentionShotAggregator()(x)
    assert tuple(attn_out["aggregated_feature"].shape) == (2, 16)
    assert torch.allclose(attn_out["attention_weights"].sum(dim=1), torch.ones(2), atol=1e-5)

    geo = torch.randn(2, 4, 6)
    source_out = SourceAwareAttentionAggregator()(x, source_geometry=geo)
    assert tuple(source_out["aggregated_feature"].shape) == (2, 16)
    assert torch.allclose(source_out["attention_weights"].sum(dim=1), torch.ones(2), atol=1e-5)


def test_source_aware_aggregator_fallbacks_without_geometry():
    from fwi_visionfm.models.aggregators import SourceAwareAttentionAggregator
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    x = torch.randn(1, 3, 8, 4, 4)
    out = SourceAwareAttentionAggregator()(x)
    assert tuple(out["aggregated_feature"].shape) == (1, 8, 4, 4)
    assert tuple(out["attention_weights"].shape) == (1, 3)

