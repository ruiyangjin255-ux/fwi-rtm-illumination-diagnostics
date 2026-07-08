from __future__ import annotations


def test_loss_registry_returns_total_and_components():
    from fwi_visionfm.training.losses import compute_loss_components
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    prediction = torch.ones(2, 1, 6, 7)
    target = torch.zeros(2, 1, 6, 7)
    losses = compute_loss_components(
        prediction,
        target,
        weights={
            "l1": 1.0,
            "gradient_l1": 0.2,
            "laplacian_l1": 0.1,
            "edge_weighted_l1": 0.3,
            "tv_smoothness": 0.05,
        },
    )
    assert "total_loss" in losses
    for key in ("l1", "gradient_l1", "laplacian_l1", "edge_weighted_l1", "tv_smoothness"):
        assert key in losses


def test_gradient_and_laplacian_losses_run_on_toy_tensor():
    from fwi_visionfm.training.losses import gradient_l1_loss, laplacian_l1_loss
    from fwi_visionfm.torch_backend import require_torch_backend

    torch = require_torch_backend()
    target = torch.linspace(0.0, 1.0, 6 * 7).view(1, 1, 6, 7)
    prediction = target.roll(1, dims=-1)
    assert float(gradient_l1_loss(prediction, target).detach().cpu()) >= 0.0
    assert float(laplacian_l1_loss(prediction, target).detach().cpu()) >= 0.0
