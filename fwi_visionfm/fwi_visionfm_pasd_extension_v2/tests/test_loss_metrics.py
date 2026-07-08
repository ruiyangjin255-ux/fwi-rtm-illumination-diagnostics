import torch

from fwi_visionfm.pasd.losses import BackgroundEdgeLoss
from fwi_visionfm.pasd.metrics import per_sample_metrics


def test_loss_and_metrics_are_finite():
    prediction = torch.rand(3, 1, 24, 24, requires_grad=True)
    background = torch.rand(3, 1, 24, 24, requires_grad=True)
    target = torch.rand(3, 1, 24, 24)
    loss = BackgroundEdgeLoss()(prediction, background, target)
    loss.total.backward()
    assert torch.isfinite(loss.total)
    values = per_sample_metrics(prediction.detach(), target)
    assert set(values) == {"mae", "rmse", "ssim", "edge_mae", "gradient_error"}
    assert all(torch.isfinite(value).all() for value in values.values())
