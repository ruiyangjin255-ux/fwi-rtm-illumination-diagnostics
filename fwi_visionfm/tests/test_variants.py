import torch

from fwi_visionfm.pasd.experiment import TrainingConfig, build_criterion, build_model
from fwi_visionfm.pasd.losses import BackgroundEdgeLoss, VelocityL1Loss
from fwi_visionfm.pasd.registry import get_variant


def test_b1_and_b4_are_architecturally_distinct_controls():
    config = TrainingConfig(base_channels=4, latent_channels=16, latent_size=(3, 3))
    b1 = get_variant("B1_raw_unet")
    b4 = get_variant("B4_pasd_fwi")
    model_b1 = build_model(b1, (16, 16), config)
    model_b4 = build_model(b4, (16, 16), config)
    assert model_b1.bridge_mode == "raw"
    assert model_b1.decoder_mode == "plain"
    assert model_b1.encoder.stem.block[0].in_channels == 1
    assert model_b4.bridge_mode == "hybrid"
    assert model_b4.decoder_mode == "decoupled"
    assert model_b4.encoder.stem.block[0].in_channels == 3
    assert isinstance(build_criterion(b1, config), VelocityL1Loss)
    assert isinstance(build_criterion(b4, config), BackgroundEdgeLoss)
    records = torch.randn(1, 3, 24, 8)
    assert model_b1(records).velocity.shape == model_b4(records).velocity.shape == (1, 1, 16, 16)
