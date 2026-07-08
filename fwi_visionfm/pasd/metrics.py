"""Numerical and structural metrics computed on physical velocity units."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor

from .losses import gradient_magnitude


def _as_4d(x: Tensor) -> Tensor:
    if x.ndim == 3:
        x = x.unsqueeze(1)
    if x.ndim != 4:
        raise ValueError("Metric tensors must be [B,H,W] or [B,C,H,W].")
    return x.float()


def _ssim_global(prediction: Tensor, target: Tensor) -> Tensor:
    """A deterministic global SSIM proxy with per-sample dynamic-range stabilization."""

    prediction = _as_4d(prediction)
    target = _as_4d(target)
    mu_x = prediction.mean(dim=(-2, -1), keepdim=True)
    mu_y = target.mean(dim=(-2, -1), keepdim=True)
    var_x = ((prediction - mu_x).square()).mean(dim=(-2, -1), keepdim=True)
    var_y = ((target - mu_y).square()).mean(dim=(-2, -1), keepdim=True)
    cov = ((prediction - mu_x) * (target - mu_y)).mean(dim=(-2, -1), keepdim=True)
    joint_min = torch.minimum(prediction.amin(dim=(-2, -1), keepdim=True), target.amin(dim=(-2, -1), keepdim=True))
    joint_max = torch.maximum(prediction.amax(dim=(-2, -1), keepdim=True), target.amax(dim=(-2, -1), keepdim=True))
    data_range = (joint_max - joint_min).clamp_min(1e-6)
    c1 = (0.01 * data_range).square()
    c2 = (0.03 * data_range).square()
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2))
    return ssim.flatten(start_dim=1).mean(dim=1)


def per_sample_metrics(prediction: Tensor, target: Tensor, edge_quantile: float = 0.8) -> Dict[str, Tensor]:
    """Return one aligned metric tensor for every sample in the batch."""

    if not 0.0 < edge_quantile < 1.0:
        raise ValueError("edge_quantile must be in (0, 1).")
    prediction = _as_4d(prediction)
    target = _as_4d(target)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target metric tensors must match.")
    error = prediction - target
    mae = error.abs().flatten(start_dim=1).mean(dim=1)
    rmse = torch.sqrt(error.square().flatten(start_dim=1).mean(dim=1))
    ssim = _ssim_global(prediction, target)
    data_range = (target.amax(dim=(-2, -1), keepdim=True) - target.amin(dim=(-2, -1), keepdim=True)).flatten(start_dim=1).mean(dim=1).clamp_min(1e-6)
    psnr = 20.0 * torch.log10(data_range / rmse.clamp_min(1e-6))

    pred_grad = gradient_magnitude(prediction)
    target_grad = gradient_magnitude(target)
    gradient_error = (pred_grad - target_grad).abs().flatten(start_dim=1).mean(dim=1)
    pred_lap = torch.nn.functional.pad(prediction[:, :, 1:, :] - prediction[:, :, :-1, :], (0, 0, 0, 1), mode="replicate")
    pred_lap = pred_lap + torch.nn.functional.pad(prediction[:, :, :, 1:] - prediction[:, :, :, :-1], (0, 1, 0, 0), mode="replicate")
    target_lap = torch.nn.functional.pad(target[:, :, 1:, :] - target[:, :, :-1, :], (0, 0, 0, 1), mode="replicate")
    target_lap = target_lap + torch.nn.functional.pad(target[:, :, :, 1:] - target[:, :, :, :-1], (0, 1, 0, 0), mode="replicate")
    laplacian_error = (pred_lap - target_lap).abs().flatten(start_dim=1).mean(dim=1)
    relative_error = torch.linalg.vector_norm(error.flatten(start_dim=1), dim=1) / torch.linalg.vector_norm(target.flatten(start_dim=1), dim=1).clamp_min(1e-6)
    flat = target_grad.flatten(start_dim=1)
    threshold = torch.quantile(flat, edge_quantile, dim=1, keepdim=True)
    edge_mask = (flat >= threshold).reshape_as(target_grad)
    edge_error = ((prediction - target).abs() * edge_mask).flatten(start_dim=1).sum(dim=1) / edge_mask.flatten(start_dim=1).sum(dim=1).clamp_min(1)
    return {
        "mae": mae,
        "rmse": rmse,
        "ssim": ssim,
        "psnr": psnr,
        "edge_mae": edge_error,
        "gradient_error": gradient_error,
        "laplacian_error": laplacian_error,
        "relative_error": relative_error,
    }
