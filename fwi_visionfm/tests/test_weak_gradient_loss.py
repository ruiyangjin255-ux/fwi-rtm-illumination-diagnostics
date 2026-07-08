from __future__ import annotations


def test_weak_gradient_l1_preset_can_be_used_in_decoder_matrix():
    from fwi_visionfm.scripts.train_local_mae_feature_decoder import LOSS_PRESETS

    assert "weak_gradient_l1" in LOSS_PRESETS
    assert LOSS_PRESETS["weak_gradient_l1"]["l1"] == 1.0
    assert LOSS_PRESETS["weak_gradient_l1"]["gradient_l1"] == 0.02

