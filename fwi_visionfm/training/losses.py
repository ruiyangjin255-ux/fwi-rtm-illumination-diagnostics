from __future__ import annotations

from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def require_torch_backend():
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch backend is unavailable for structure-aware losses.")
    return torch


def _velocity_tensor(x: Any) -> Any:
    if isinstance(x, dict):
        if "velocity" not in x:
            raise ValueError("prediction/target dict must contain 'velocity'")
        return x["velocity"]
    return x


def _ensure_4d(x: Any) -> Any:
    x = _velocity_tensor(x)
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 4:
        return x
    raise ValueError(f"loss input must be [B,H,W] or [B,1,H,W], got {tuple(x.shape)}")


def _gradients(x: Any) -> tuple[Any, Any]:
    tensor = _ensure_4d(x)
    gx = tensor[..., :, 1:] - tensor[..., :, :-1]
    gy = tensor[..., 1:, :] - tensor[..., :-1, :]
    return gx, gy


def _laplacian(x: Any) -> Any:
    torch = require_torch_backend()
    tensor = _ensure_4d(x)
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=tensor.dtype, device=tensor.device)
    kernel = kernel.view(1, 1, 3, 3)
    return torch.nn.functional.conv2d(tensor, kernel, padding=1)


def l1_loss(prediction: Any, target: Any) -> Any:
    torch = require_torch_backend()
    return torch.nn.functional.l1_loss(_ensure_4d(prediction), _ensure_4d(target))


def mse_loss(prediction: Any, target: Any) -> Any:
    torch = require_torch_backend()
    return torch.nn.functional.mse_loss(_ensure_4d(prediction), _ensure_4d(target))


def gradient_l1_loss(prediction: Any, target: Any) -> Any:
    torch = require_torch_backend()
    pred_gx, pred_gy = _gradients(prediction)
    true_gx, true_gy = _gradients(target)
    return 0.5 * (torch.nn.functional.l1_loss(pred_gx, true_gx) + torch.nn.functional.l1_loss(pred_gy, true_gy))


def laplacian_l1_loss(prediction: Any, target: Any) -> Any:
    torch = require_torch_backend()
    return torch.nn.functional.l1_loss(_laplacian(prediction), _laplacian(target))


def edge_weighted_l1_loss(prediction: Any, target: Any) -> Any:
    torch = require_torch_backend()
    target_tensor = _ensure_4d(target)
    pred_tensor = _ensure_4d(prediction)
    grad_x, grad_y = _gradients(target_tensor)
    grad_x_full = torch.nn.functional.pad(grad_x.abs(), (0, 1, 0, 0))
    grad_y_full = torch.nn.functional.pad(grad_y.abs(), (0, 0, 0, 1))
    weight = 1.0 + 0.5 * (grad_x_full + grad_y_full)
    return ((pred_tensor - target_tensor).abs() * weight).mean()


def tv_smoothness_loss(prediction: Any, target: Any | None = None) -> Any:
    del target
    torch = require_torch_backend()
    pred_tensor = _ensure_4d(prediction)
    grad_x, grad_y = _gradients(pred_tensor)
    return 0.5 * (grad_x.abs().mean() + grad_y.abs().mean())


def boundary_aux_l1_loss(
    prediction: Any,
    target: Any,
    *,
    lambda_boundary: float = 0.05,
    boundary_method: str = "gradient_magnitude",
    threshold: float = 0.2,
) -> Any:
    from fwi_visionfm.data.boundary_targets import build_velocity_boundary_target

    torch = require_torch_backend()
    if not isinstance(prediction, dict) or "boundary" not in prediction:
        raise ValueError("boundary_aux_l1 requires model output dict with 'boundary' prediction")
    velocity_true = _ensure_4d(target)
    boundary_true = target.get("boundary") if isinstance(target, dict) else None
    if boundary_true is None:
        boundary_true = build_velocity_boundary_target(velocity_true, method=boundary_method, threshold=threshold)
    else:
        boundary_true = _ensure_4d(boundary_true)
    velocity_loss = torch.nn.functional.l1_loss(_ensure_4d(prediction["velocity"]), velocity_true)
    boundary_loss = torch.nn.functional.l1_loss(_ensure_4d(prediction["boundary"]), boundary_true)
    return velocity_loss + float(lambda_boundary) * boundary_loss


def compute_loss_components(
    prediction: Any,
    target: Any,
    *,
    weights: dict[str, float] | None = None,
    component_kwargs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    torch = require_torch_backend()
    resolved = dict(weights or {"l1": 1.0})
    kwargs_map = dict(component_kwargs or {})
    registry = {
        "l1": l1_loss,
        "mse": mse_loss,
        "gradient_l1": gradient_l1_loss,
        "laplacian_l1": laplacian_l1_loss,
        "edge_weighted_l1": edge_weighted_l1_loss,
        "tv_smoothness": tv_smoothness_loss,
        "boundary_aux_l1": boundary_aux_l1_loss,
    }
    components: dict[str, Any] = {}
    total = None
    for name, weight in resolved.items():
        if float(weight) == 0.0:
            continue
        if name not in registry:
            raise ValueError(f"unsupported loss component: {name}")
        value = registry[name](prediction, target, **kwargs_map.get(name, {}))
        components[name] = value
        weighted = value * float(weight)
        total = weighted if total is None else total + weighted
    if total is None:
        total = torch.zeros((), dtype=_ensure_4d(prediction).dtype, device=_ensure_4d(prediction).device)
    components["total_loss"] = total
    return components
