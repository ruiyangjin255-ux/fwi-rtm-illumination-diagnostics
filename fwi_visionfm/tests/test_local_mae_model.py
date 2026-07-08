from __future__ import annotations

import pytest


def test_local_mae_forward_returns_reconstruction_mask_and_loss():
    torch = pytest.importorskip("torch")

    from fwi_visionfm.models.seismic_backbones.local_mae import LocalSeismicMAE

    model = LocalSeismicMAE(
        input_size=64,
        patch_size=8,
        in_chans=3,
        embed_dim=128,
        depth=2,
        num_heads=4,
        decoder_embed_dim=64,
        decoder_depth=1,
        decoder_heads=4,
        mask_ratio=0.75,
    )
    image = torch.randn(2, 3, 64, 64)
    output = model(image)

    assert tuple(output["reconstruction"].shape) == (2, 3, 64, 64)
    assert output["mask"].shape[0] == 2
    assert output["latent_features"].shape[0] == 2
    assert float(output["reconstruction_loss"].detach().cpu()) >= 0.0

