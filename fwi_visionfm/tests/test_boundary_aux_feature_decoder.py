from __future__ import annotations


def test_boundary_aux_feature_decoder_can_forward_feature_tokens():
    from fwi_visionfm.models.decoders import build_decoder
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    decoder = build_decoder("boundary_aux_unet", output_shape=(70, 70), base_channels=16, vmin=1500.0, vmax=4500.0)
    out = decoder(torch.randn(2, 768))
    assert set(out.keys()) == {"velocity", "boundary"}
    assert tuple(out["velocity"].shape) == (2, 1, 70, 70)
    assert tuple(out["boundary"].shape) == (2, 1, 70, 70)
