from __future__ import annotations

from typing import Any

from fwi_visionfm.torch_backend import require_torch_backend


def _ensure_4d(velocity: Any) -> Any:
    if velocity.ndim == 3:
        return velocity.unsqueeze(1)
    if velocity.ndim == 4:
        return velocity
    raise ValueError(f"velocity must be [B,H,W] or [B,1,H,W], got {tuple(velocity.shape)}")


def _normalize01(x: Any) -> Any:
    xmin = x.amin(dim=(-2, -1), keepdim=True)
    xmax = x.amax(dim=(-2, -1), keepdim=True)
    return (x - xmin) / (xmax - xmin).clamp_min(1.0e-6)


def build_velocity_boundary_target(velocity: Any, method: str = "gradient_magnitude", threshold: float = 0.2) -> Any:
    torch = require_torch_backend()
    x = _ensure_4d(velocity).float()
    if method == "gradient_magnitude" or method == "thresholded_gradient":
        grad_x = torch.nn.functional.pad((x[..., :, 1:] - x[..., :, :-1]).abs(), (0, 1, 0, 0))
        grad_y = torch.nn.functional.pad((x[..., 1:, :] - x[..., :-1, :]).abs(), (0, 0, 0, 1))
        boundary = _normalize01(torch.sqrt(grad_x.square() + grad_y.square() + 1.0e-12))
        if method == "thresholded_gradient":
            boundary = (boundary >= float(threshold)).float()
    elif method == "sobel":
        kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        gx = torch.nn.functional.conv2d(x, kernel_x, padding=1)
        gy = torch.nn.functional.conv2d(x, kernel_y, padding=1)
        boundary = _normalize01(torch.sqrt(gx.square() + gy.square() + 1.0e-12))
    elif method == "laplacian":
        kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
        boundary = _normalize01(torch.nn.functional.conv2d(x, kernel, padding=1).abs())
    else:
        raise ValueError(f"unsupported boundary method: {method}")
    return boundary
