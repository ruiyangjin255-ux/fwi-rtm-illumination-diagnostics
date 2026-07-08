"""Losses for baseline and background-edge decoupled velocity regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _as_velocity_4d(x: Tensor) -> Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(1)
    if x.ndim != 4 or x.shape[1] != 1:
        raise ValueError("velocity tensors must have shape [B,H,W] or [B,1,H,W].")
    return x.float()


def gradients_xy(x: Tensor) -> Tuple[Tensor, Tensor]:
    x = _as_velocity_4d(x)
    gx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0), mode="replicate")
    gz = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1), mode="replicate")
    return gx, gz


def gradient_magnitude(x: Tensor) -> Tensor:
    gx, gz = gradients_xy(x)
    return torch.sqrt(gx.square() + gz.square() + 1e-12)


def gaussian_blur2d(x: Tensor, sigma: float = 1.5) -> Tensor:
    x = _as_velocity_4d(x)
    if sigma <= 0.0:
        return x
    radius = max(1, int(round(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel_1d = torch.exp(-(coords.square()) / (2.0 * sigma * sigma))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d).reshape(1, 1, 2 * radius + 1, 2 * radius + 1)
    return F.conv2d(x, kernel_2d, padding=radius)


@dataclass
class LossOutput:
    total: Tensor
    components: Dict[str, Tensor]
    background_target: Tensor
    edge_mask: Tensor


class VelocityCriterion(Protocol):
    def __call__(self, prediction: Tensor, background_prediction: Tensor, target: Tensor) -> LossOutput: ...


class VelocityL1Loss(nn.Module):
    """Ordinary velocity L1 control loss used for B1/B2."""

    def forward(self, prediction: Tensor, background_prediction: Tensor, target: Tensor) -> LossOutput:
        prediction = _as_velocity_4d(prediction)
        target = _as_velocity_4d(target)
        l1 = F.l1_loss(prediction, target)
        edge_mask = torch.zeros_like(target)
        return LossOutput(
            total=l1,
            components={"l1": l1.detach(), "background": l1.detach() * 0.0, "edge": l1.detach() * 0.0, "smooth": l1.detach() * 0.0},
            background_target=target,
            edge_mask=edge_mask,
        )


class BackgroundEdgeLoss(nn.Module):
    """Stabilize background velocity and preserve true interfaces without global edge sharpening."""

    def __init__(
        self,
        background_sigma: float = 1.5,
        edge_quantile: float = 0.8,
        weight_l1: float = 1.0,
        weight_background: float = 0.25,
        weight_edge: float = 0.1,
        weight_smooth: float = 0.02,
    ) -> None:
        super().__init__()
        if not 0.0 < edge_quantile < 1.0:
            raise ValueError("edge_quantile must be between 0 and 1.")
        self.background_sigma = float(background_sigma)
        self.edge_quantile = float(edge_quantile)
        self.weight_l1 = float(weight_l1)
        self.weight_background = float(weight_background)
        self.weight_edge = float(weight_edge)
        self.weight_smooth = float(weight_smooth)

    def forward(self, prediction: Tensor, background_prediction: Tensor, target: Tensor) -> LossOutput:
        prediction = _as_velocity_4d(prediction)
        background_prediction = _as_velocity_4d(background_prediction)
        target = _as_velocity_4d(target)
        if prediction.shape != target.shape:
            raise ValueError("prediction and target shapes must match.")

        background_target = gaussian_blur2d(target, sigma=self.background_sigma)
        target_grad = gradient_magnitude(target)
        flat = target_grad.flatten(start_dim=1)
        threshold = torch.quantile(flat, self.edge_quantile, dim=1, keepdim=True)
        edge_mask = (flat >= threshold).reshape_as(target_grad).float()

        l1 = F.l1_loss(prediction, target)
        bg = F.l1_loss(background_prediction, background_target)

        pred_gx, pred_gz = gradients_xy(prediction)
        target_gx, target_gz = gradients_xy(target)
        gradient_difference = (pred_gx - target_gx).abs() + (pred_gz - target_gz).abs()
        edge = (edge_mask * gradient_difference).sum() / edge_mask.sum().clamp_min(1.0)

        non_edge = 1.0 - edge_mask
        smooth = (non_edge * gradient_magnitude(prediction)).sum() / non_edge.sum().clamp_min(1.0)

        total = (
            self.weight_l1 * l1
            + self.weight_background * bg
            + self.weight_edge * edge
            + self.weight_smooth * smooth
        )
        components = {"l1": l1.detach(), "background": bg.detach(), "edge": edge.detach(), "smooth": smooth.detach()}
        return LossOutput(total=total, components=components, background_target=background_target, edge_mask=edge_mask)
