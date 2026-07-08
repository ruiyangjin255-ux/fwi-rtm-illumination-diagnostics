from __future__ import annotations


def test_boundary_aux_decoder_can_forward():
    from fwi_visionfm.models.decoders import build_decoder
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    decoder = build_decoder("boundary_aux_unet", output_shape=(64, 64), base_channels=16, vmin=1500.0, vmax=4500.0)
    out = decoder(torch.randn(2, 196, 32))
    assert set(out.keys()) == {"velocity", "boundary"}
    assert tuple(out["velocity"].shape) == (2, 1, 64, 64)
    assert tuple(out["boundary"].shape) == (2, 1, 64, 64)

