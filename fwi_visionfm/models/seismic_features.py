from __future__ import annotations

from fwi_visionfm.optional_deps import missing_dependencies


def _require_torch():
    if missing_dependencies("torch"):
        raise RuntimeError(
            "PyTorch backend is unavailable. Install PyTorch first, then rerun this experiment. "
            "Suggested CPU install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
        )
    import torch

    return torch


def normalize_feature(x, mode: str = "zscore", eps: float = 1.0e-6):
    torch = _require_torch()
    x = x.to(dtype=torch.float32)
    if mode == "none":
        return x
    if mode == "zscore":
        mean = x.mean(dim=(-1, -2), keepdim=True)
        std = x.std(dim=(-1, -2), keepdim=True, unbiased=False).clamp_min(eps)
        return (x - mean) / std
    if mode == "minmax":
        min_value = x.amin(dim=(-1, -2), keepdim=True)
        max_value = x.amax(dim=(-1, -2), keepdim=True)
        scale = (max_value - min_value).clamp_min(eps)
        return (x - min_value) / scale
    raise ValueError(f"unsupported normalization mode: {mode}")


def resize_feature(x, image_size: int):
    torch = _require_torch()
    resized = torch.nn.functional.interpolate(
        x.unsqueeze(1).to(dtype=torch.float32),
        size=(int(image_size), int(image_size)),
        mode="bilinear",
        align_corners=False,
    )
    return resized[:, 0]


def compute_envelope(x, dim: int = -2):
    torch = _require_torch()
    x = x.to(dtype=torch.float32)
    n = x.shape[dim]
    spectrum = torch.fft.fft(x, dim=dim)
    h = torch.zeros(n, dtype=spectrum.real.dtype, device=x.device)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1:n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1:(n + 1) // 2] = 2.0
    shape = [1] * x.ndim
    shape[dim] = n
    analytic = torch.fft.ifft(spectrum * h.view(*shape), dim=dim)
    envelope = torch.abs(analytic).to(dtype=torch.float32)
    return torch.nan_to_num(envelope, nan=0.0, posinf=0.0, neginf=0.0)


def compute_spectrogram_energy(
    x,
    *,
    n_fft: int = 64,
    hop_length: int = 16,
    win_length: int = 64,
    power: float = 1.0,
):
    torch = _require_torch()
    x = x.to(dtype=torch.float32)
    if x.ndim != 3:
        raise ValueError(f"expected [N, T, R], got {tuple(x.shape)}")
    shots, time, receivers = x.shape
    traces = x.permute(0, 2, 1).reshape(shots * receivers, time)
    effective_n_fft = max(8, min(int(n_fft), int(time)))
    effective_win_length = max(8, min(int(win_length), effective_n_fft))
    effective_hop_length = max(1, min(int(hop_length), effective_win_length))
    window = torch.hann_window(effective_win_length, device=x.device, dtype=x.dtype)
    spec = torch.stft(
        traces,
        n_fft=effective_n_fft,
        hop_length=effective_hop_length,
        win_length=effective_win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    magnitude = spec.abs()
    if float(power) != 1.0:
        magnitude = magnitude.pow(float(power))
    energy = torch.log1p(magnitude.mean(dim=1))
    energy = energy.view(shots, receivers, energy.shape[-1]).permute(0, 2, 1).contiguous()
    return torch.nan_to_num(energy, nan=0.0, posinf=0.0, neginf=0.0)
