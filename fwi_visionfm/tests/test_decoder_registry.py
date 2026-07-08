from __future__ import annotations


def test_simple_bounded_decoder_outputs_b1hw():
    from fwi_visionfm.models.decoders import build_decoder
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    decoder = build_decoder("simple_bounded_decoder", output_shape=(70, 70), base_channels=16, vmin=1500.0, vmax=4500.0)
    x = torch.randn(2, 64)
    y = decoder(x)
    assert tuple(y.shape) == (2, 1, 70, 70)


def test_unet_decoder_outputs_b1hw():
    from fwi_visionfm.models.decoders import build_decoder
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    decoder = build_decoder("unet_decoder", output_shape=(70, 70), base_channels=16, vmin=1500.0, vmax=4500.0)
    x = torch.randn(2, 196, 32)
    y = decoder(x)
    assert tuple(y.shape) == (2, 1, 70, 70)
