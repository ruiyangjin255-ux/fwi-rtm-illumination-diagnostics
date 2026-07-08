from __future__ import annotations

import pytest


def test_boundary_aux_loss_can_be_constructed_and_generates_target():
    from fwi_visionfm.training.losses import compute_loss_components
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    prediction = {"velocity": torch.ones(2, 1, 6, 7), "boundary": torch.zeros(2, 1, 6, 7)}
    target = torch.zeros(2, 1, 6, 7)
    losses = compute_loss_components(
        prediction,
        target,
        weights={"boundary_aux_l1": 1.0},
        component_kwargs={"boundary_aux_l1": {"lambda_boundary": 0.05, "boundary_method": "gradient_magnitude"}},
    )
    assert "boundary_aux_l1" in losses
    assert "total_loss" in losses


def test_boundary_aux_loss_requires_boundary_prediction():
    from fwi_visionfm.training.losses import boundary_aux_l1_loss
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    with pytest.raises(ValueError):
        boundary_aux_l1_loss({"velocity": torch.ones(1, 1, 4, 4)}, torch.zeros(1, 1, 4, 4))

