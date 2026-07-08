"""Physics-aligned multi-attribute bridge for shot-gather FWI inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class BridgeOutput:
    """Bridge output.

    attributes: [B, S, 3, T, R] = raw, envelope, band-energy.
    geometry:   [B, S, 3]       = source, normalized shot index, mean normalized offset.
    """

    attributes: Tensor
    geometry: Tensor


def _as_shot_tensor(records: Tensor) -> Tensor:
    """Validate and normalize accepted record layouts to [B, S, T, R]."""
    if records.ndim == 5 and records.shape[2] == 1:
        records = records[:, :, 0]
    if records.ndim != 4:
        raise ValueError(
            "Expected seismic records with shape [B,S,T,R] or [B,S,1,T,R], "
            f"received {tuple(records.shape)}."
        )
    return records.float()


def _hilbert_envelope(x: Tensor, dim: int = -2) -> Tensor:
    """Compute Hilbert-envelope along the temporal dimension using torch FFT."""
    n = x.shape[dim]
    spectrum = torch.fft.fft(x, dim=dim)
    h = torch.zeros(n, device=x.device, dtype=spectrum.real.dtype)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    shape = [1] * x.ndim
    shape[dim] = n
    analytic = torch.fft.ifft(spectrum * h.reshape(shape), dim=dim)
    return analytic.abs()


def _temporal_lowpass(x: Tensor, kernel_size: int) -> Tensor:
    """Apply a lightweight time-axis moving average to [B,S,T,R] gathers."""
    if kernel_size < 3:
        return x
    if kernel_size % 2 == 0:
        kernel_size += 1
    b, s, t, r = x.shape
    # Convert each receiver trace into a 1-D signal: [B*S*R, 1, T].
    traces = x.permute(0, 1, 3, 2).reshape(b * s * r, 1, t)
    low = F.avg_pool1d(traces, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return low.reshape(b, s, r, t).permute(0, 1, 3, 2)


class HybridAttributeBridge(nn.Module):
    """Construct raw, envelope, and band-energy gather attributes.

    The bridge is deliberately lightweight. It preserves waveform morphology, emphasizes
    reflection energy, and supplies a low/high-frequency proxy without STFT overhead.
    """

    def __init__(
        self,
        lowpass_kernel: int = 21,
        band_low_weight: float = 0.5,
        robust_clip_quantile: float = 0.995,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not 0.0 < robust_clip_quantile <= 1.0:
            raise ValueError("robust_clip_quantile must be within (0, 1].")
        self.lowpass_kernel = int(lowpass_kernel)
        self.band_low_weight = float(band_low_weight)
        self.robust_clip_quantile = float(robust_clip_quantile)
        self.eps = float(eps)

    def _robust_normalize(self, x: Tensor) -> Tensor:
        b, s, t, r = x.shape
        flattened = x.abs().reshape(b, s, -1)
        q = torch.quantile(flattened, self.robust_clip_quantile, dim=-1, keepdim=True)
        q = q.clamp_min(self.eps).reshape(b, s, 1, 1)
        clipped = x.clamp(min=-q, max=q)
        mean = clipped.mean(dim=(-2, -1), keepdim=True)
        std = clipped.std(dim=(-2, -1), keepdim=True).clamp_min(self.eps)
        return (clipped - mean) / std

    @staticmethod
    def _geometry(
        records: Tensor,
        source_positions: Optional[Tensor],
        receiver_positions: Optional[Tensor],
    ) -> Tensor:
        b, s, _, r = records.shape
        device, dtype = records.device, records.dtype
        shot_index = torch.linspace(-1.0, 1.0, s, device=device, dtype=dtype).reshape(1, s, 1)
        shot_index = shot_index.expand(b, -1, -1)

        if source_positions is None:
            source = shot_index.squeeze(-1)
        else:
            source = source_positions.to(device=device, dtype=dtype)
            if source.ndim == 1:
                source = source.unsqueeze(0).expand(b, -1)
            if source.shape != (b, s):
                raise ValueError("source_positions must have shape [S] or [B,S].")
            source_min = source.amin(dim=1, keepdim=True)
            source_range = (source.amax(dim=1, keepdim=True) - source_min).clamp_min(1e-6)
            source = 2.0 * (source - source_min) / source_range - 1.0

        if receiver_positions is None:
            receivers = torch.linspace(-1.0, 1.0, r, device=device, dtype=dtype).reshape(1, 1, r)
            receivers = receivers.expand(b, s, -1)
        else:
            receivers = receiver_positions.to(device=device, dtype=dtype)
            if receivers.ndim == 1:
                receivers = receivers.reshape(1, 1, r).expand(b, s, -1)
            elif receivers.ndim == 2:
                receivers = receivers.unsqueeze(1).expand(-1, s, -1)
            if receivers.shape != (b, s, r):
                raise ValueError("receiver_positions must have shape [R], [B,R], or [B,S,R].")
            rmin = receivers.amin(dim=-1, keepdim=True)
            rrange = (receivers.amax(dim=-1, keepdim=True) - rmin).clamp_min(1e-6)
            receivers = 2.0 * (receivers - rmin) / rrange - 1.0

        mean_offset = (receivers - source.unsqueeze(-1)).abs().mean(dim=-1)
        return torch.stack([source, shot_index.squeeze(-1), mean_offset], dim=-1)

    def forward(
        self,
        records: Tensor,
        source_positions: Optional[Tensor] = None,
        receiver_positions: Optional[Tensor] = None,
    ) -> BridgeOutput:
        gathers = _as_shot_tensor(records)
        raw = self._robust_normalize(gathers)
        envelope = _hilbert_envelope(raw)
        envelope = self._robust_normalize(envelope)

        low = _temporal_lowpass(raw, self.lowpass_kernel)
        high = raw - low
        band_energy = self.band_low_weight * low.abs() + (1.0 - self.band_low_weight) * high.abs()
        band_energy = self._robust_normalize(band_energy)

        attributes = torch.stack([raw, envelope, band_energy], dim=2)
        geometry = self._geometry(gathers, source_positions, receiver_positions)
        return BridgeOutput(attributes=attributes, geometry=geometry)
