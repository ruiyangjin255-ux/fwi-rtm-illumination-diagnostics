from __future__ import annotations


def test_shot_global_context_encoder_pools_only_valid_traces():
    from fwi_visionfm.models.geometry_aware_trace_bridge import ShotGlobalContextEncoder
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    trace_context = torch.tensor(
        [[[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]]],
        dtype=torch.float32,
    )
    valid_trace_mask = torch.tensor([[[1.0, 1.0, 0.0]]], dtype=torch.float32)
    encoder = ShotGlobalContextEncoder(input_dim=2, output_dim=4)
    output = encoder(trace_context, valid_trace_mask=valid_trace_mask)
    assert tuple(output.shape) == (1, 1, 4)
    assert output.abs().sum().item() > 0.0


def test_shot_global_context_encoder_zeroes_invalid_shots():
    from fwi_visionfm.models.geometry_aware_trace_bridge import ShotGlobalContextEncoder
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    trace_context = torch.randn(1, 2, 3, 5)
    valid_trace_mask = torch.ones(1, 2, 3)
    valid_shot_mask = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    encoder = ShotGlobalContextEncoder(input_dim=5, output_dim=6)
    output = encoder(trace_context, valid_trace_mask=valid_trace_mask, valid_shot_mask=valid_shot_mask)
    assert tuple(output.shape) == (1, 2, 6)
    assert output[0, 1].abs().sum().item() == 0.0
