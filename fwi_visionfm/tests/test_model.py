import torch

from fwi_visionfm.pasd.model import PASDFWI


def test_pasd_model_output_shapes():
    model = PASDFWI(output_size=(32, 32), base_channels=8, latent_channels=32, latent_size=(5, 5))
    records = torch.randn(2, 3, 64, 16)
    output = model(records)
    assert output.velocity.shape == (2, 1, 32, 32)
    assert output.background.shape == (2, 1, 32, 32)
    assert output.residual.shape == (2, 1, 32, 32)
    assert output.attention.shape == (2, 3)
    assert torch.allclose(output.attention.sum(dim=1), torch.ones(2), atol=1e-6)
