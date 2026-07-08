from __future__ import annotations


def test_trace_neighborhood_encoder_replicate_padding_preserves_edges():
    from fwi_visionfm.models.geometry_aware_trace_bridge import TraceNeighborhoodEncoder
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    records = torch.arange(1 * 1 * 3 * 4, dtype=torch.float32).reshape(1, 1, 3, 4)
    encoder = TraceNeighborhoodEncoder(trace_context_radius=1, padding_mode="replicate", output_dim=8)
    neighborhood = encoder.build_neighborhood(records)
    assert tuple(neighborhood.shape) == (1, 1, 3, 3, 4)
    assert neighborhood[0, 0, 0, 0].tolist() == records[0, 0, 0].tolist()
    assert neighborhood[0, 0, 2, 2].tolist() == records[0, 0, 2].tolist()


def test_trace_neighborhood_encoder_zero_padding_sets_outer_neighbors_to_zero():
    from fwi_visionfm.models.geometry_aware_trace_bridge import TraceNeighborhoodEncoder
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    records = torch.ones((1, 1, 2, 3), dtype=torch.float32)
    encoder = TraceNeighborhoodEncoder(trace_context_radius=1, padding_mode="zero", output_dim=4)
    neighborhood = encoder.build_neighborhood(records)
    assert neighborhood[0, 0, 0, 0].sum().item() == 0.0
    assert neighborhood[0, 0, 1, 2].sum().item() == 0.0


def test_trace_neighborhood_encoder_outputs_trace_level_features():
    from fwi_visionfm.models.geometry_aware_trace_bridge import TraceNeighborhoodEncoder
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    records = torch.randn(2, 3, 5, 16)
    encoder = TraceNeighborhoodEncoder(trace_context_radius=2, output_dim=12)
    output = encoder(records)
    assert tuple(output.shape) == (2, 3, 5, 12)
