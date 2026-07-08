from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fwi_visionfm.torch_backend import require_torch_backend


BRIDGE_NAMES = [
    "raw_repeat3",
    "raw_normalized3",
    "envelope_repeat3",
    "raw_plus_envelope",
    "raw_spectrogram",
    "spectrogram_multiband",
    "raw_envelope_spectrum3",
    "low_high_frequency_bridge",
]


def _as_torch(records: Any):
    torch = require_torch_backend()
    if isinstance(records, np.ndarray):
        return torch.as_tensor(records, dtype=torch.float32), True
    return records.float(), False


def _normalize(x):
    return x / x.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)


def _zscore(x):
    return (x - x.mean(dim=(-2, -1), keepdim=True)) / x.std(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)


def _envelope(x):
    torch = require_torch_backend()
    spectrum = torch.fft.fft(x, dim=-1)
    n = x.shape[-1]
    h = torch.zeros(n, dtype=x.dtype, device=x.device)
    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[0] = 1
        h[1 : (n + 1) // 2] = 2
    analytic = torch.fft.ifft(spectrum * h.view(*([1] * (x.ndim - 1)), n), dim=-1)
    return analytic.abs()


def _band_energy(x, bands: int = 3):
    torch = require_torch_backend()
    mag = torch.fft.rfft(x, dim=-1).abs()
    chunks = torch.chunk(mag, bands, dim=-1)
    maps = [chunk.mean(dim=-1, keepdim=True).expand(*x.shape[:-1], x.shape[-1]) for chunk in chunks]
    while len(maps) < bands:
        maps.append(torch.zeros_like(x))
    return maps[:bands]


def _low_high(x):
    torch = require_torch_backend()
    freq = torch.fft.rfft(x, dim=-1)
    cutoff = max(1, freq.shape[-1] // 4)
    low_freq = torch.zeros_like(freq)
    high_freq = torch.zeros_like(freq)
    low_freq[..., :cutoff] = freq[..., :cutoff]
    high_freq[..., cutoff:] = freq[..., cutoff:]
    low = torch.fft.irfft(low_freq, n=x.shape[-1], dim=-1)
    high = torch.fft.irfft(high_freq, n=x.shape[-1], dim=-1)
    return low, high


def _resize(image, output_size: tuple[int, int] | None):
    if output_size is None:
        return image
    torch = require_torch_backend()
    if image.ndim == 3:
        return torch.nn.functional.interpolate(image.unsqueeze(0), size=output_size, mode="bilinear", align_corners=False)[0]
    return torch.nn.functional.interpolate(image, size=output_size, mode="bilinear", align_corners=False)


@dataclass
class SeismicBridge:
    name: str
    config: dict[str, Any]

    def __post_init__(self) -> None:
        if self.name not in BRIDGE_NAMES:
            raise ValueError(f"unsupported bridge: {self.name}")
        output_size = self.config.get("output_size", [64, 64])
        self.output_size = tuple(int(v) for v in output_size) if output_size is not None else None
        self.last_metadata: dict[str, Any] = {}

    @property
    def output_channels(self) -> int:
        return 3

    def _collapse_shots(self, records):
        if records.ndim == 3:
            return records.mean(dim=0), False
        if records.ndim == 4:
            return records.mean(dim=1), True
        raise ValueError(f"records must have shape (shots,time,receivers) or (batch,shots,time,receivers), got {tuple(records.shape)}")

    def _channels(self, raw):
        normalized = _normalize(raw)
        zscored = _zscore(raw)
        envelope = _normalize(_envelope(raw))
        abs_raw = _normalize(raw.abs())
        bands = [_normalize(band) for band in _band_energy(raw, 3)]
        low, high = _low_high(raw)
        low = _normalize(low)
        high = _normalize(high.abs())
        if self.name == "raw_repeat3":
            return [normalized, normalized, normalized], "raw"
        if self.name == "raw_normalized3":
            return [zscored, zscored, zscored], "zscore_raw"
        if self.name == "envelope_repeat3":
            return [envelope, envelope, envelope], "envelope"
        if self.name == "raw_plus_envelope":
            return [normalized, envelope, abs_raw], "raw_envelope_abs"
        if self.name == "raw_spectrogram":
            return [bands[0], bands[1], bands[2]], "fft_spectrogram"
        if self.name == "spectrogram_multiband":
            return [bands[0], bands[1], bands[2]], "fft_multiband"
        if self.name == "raw_envelope_spectrum3":
            return [normalized, envelope, _normalize(sum(bands) / 3.0)], "raw_envelope_spectrum"
        if self.name == "low_high_frequency_bridge":
            return [low, high, normalized], "low_high_raw"
        raise ValueError(f"unsupported bridge: {self.name}")

    def forward(self, records: Any) -> dict[str, Any]:
        tensor, was_numpy = _as_torch(records)
        input_shape = list(tensor.shape)
        raw, batched = self._collapse_shots(tensor)
        channels, attribute_type = self._channels(raw)
        image = require_torch_backend().stack(channels, dim=1 if batched else 0)
        image = _resize(image, self.output_size)
        metadata = {
            "input_shape": input_shape,
            "output_shape": list(image.shape),
            "normalization": "per-sample",
            "attribute_type": attribute_type,
            "frequency_info": "rfft-based attributes" if "spectrum" in attribute_type or "fft" in attribute_type or "low_high" in attribute_type else "",
        }
        self.last_metadata = metadata
        if was_numpy:
            image_out = image.detach().cpu().numpy().astype(np.float32)
        else:
            image_out = image
        return {"image": image_out, "bridge_name": self.name, "metadata": metadata}


def build_bridge(name: str, config: dict[str, Any] | None = None) -> SeismicBridge:
    return SeismicBridge(str(name), dict(config or {}))
