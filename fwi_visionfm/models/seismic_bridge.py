from __future__ import annotations

from typing import Literal

from fwi_visionfm.optional_deps import missing_dependencies
from fwi_visionfm.models.seismic_features import (
    compute_envelope,
    compute_spectrogram_energy,
    normalize_feature,
    resize_feature,
)


def _require_torch():
    if missing_dependencies("torch"):
        raise RuntimeError(
            "PyTorch backend is unavailable. Install PyTorch first, then rerun this experiment. "
            "Suggested CPU install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
        )
    import torch

    return torch


class SeismicToVisionBridge:
    """
    Convert multi-shot seismic records into pseudo-images for vision backbones.

    Supported input layouts:
    - [B, S, T, R]
    - [B, S, R, T]
    - [B, S, C, T, R]
    - [B, S, C, R, T]
    """

    def __init__(
        self,
        image_size: int = 224,
        in_chans: int = 3,
        norm_mode: Literal["none", "zscore", "minmax"] = "zscore",
        eps: float = 1.0e-6,
        input_layout: str = "auto",
        feature_mode: str = "raw_repeat3",
        spectrogram_n_fft: int = 64,
        spectrogram_hop_length: int = 16,
        spectrogram_win_length: int = 64,
        spectrogram_power: float = 1.0,
    ) -> None:
        self.image_size = int(image_size)
        self.in_chans = int(in_chans)
        self.norm_mode = str(norm_mode)
        self.eps = float(eps)
        self.input_layout = str(input_layout)
        self.feature_mode = str(feature_mode)
        self.spectrogram_n_fft = int(spectrogram_n_fft)
        self.spectrogram_hop_length = int(spectrogram_hop_length)
        self.spectrogram_win_length = int(spectrogram_win_length)
        self.spectrogram_power = float(spectrogram_power)
        if self.in_chans <= 0:
            raise ValueError("in_chans must be positive")
        if self.norm_mode not in {"none", "zscore", "minmax"}:
            raise ValueError(f"unsupported norm_mode: {self.norm_mode}")
        if self.feature_mode not in {
            "raw_repeat3",
            "raw_envelope_spectrogram",
            "raw_envelope_spectrum3",
            "raw_envelope",
            "raw_spectrogram",
            "envelope_repeat3",
            "spectrogram_repeat3",
            "spectrogram_multiband",
        }:
            raise ValueError(f"unsupported feature_mode: {self.feature_mode}")

    def _canonicalize(self, x):
        torch = _require_torch()
        if x.ndim == 4:
            if self.input_layout == "bstr":
                x = x.unsqueeze(2)
            elif self.input_layout == "bsrt":
                x = x.transpose(-1, -2).unsqueeze(2)
            elif self.input_layout == "auto":
                if x.shape[-1] >= x.shape[-2]:
                    x = x.transpose(-1, -2).unsqueeze(2)
                else:
                    x = x.unsqueeze(2)
            else:
                raise ValueError(f"unsupported input_layout for 4D tensor: {self.input_layout}")
        elif x.ndim == 5:
            if self.input_layout == "bsctr":
                pass
            elif self.input_layout == "bscrt":
                x = x.transpose(-1, -2)
            elif self.input_layout == "auto":
                if x.shape[-1] >= x.shape[-2]:
                    x = x.transpose(-1, -2)
            else:
                raise ValueError(f"unsupported input_layout for 5D tensor: {self.input_layout}")
        else:
            raise ValueError(f"expected 4D or 5D seismic tensor, got shape {tuple(x.shape)}")
        if x.ndim != 5:
            raise ValueError(f"failed to canonicalize seismic tensor, got shape {tuple(x.shape)}")
        if x.shape[2] <= 0:
            raise ValueError("channel dimension must be positive")
        return x.to(dtype=torch.float32)

    def _normalize(self, x):
        return normalize_feature(x, mode=self.norm_mode, eps=self.eps)

    def _adapt_channels(self, x):
        torch = _require_torch()
        if self.in_chans == x.shape[1]:
            return x
        if self.in_chans == 1:
            return x[:, :1]
        if self.in_chans < x.shape[1]:
            return x[:, : self.in_chans]
        repeats = (self.in_chans + x.shape[1] - 1) // x.shape[1]
        return x.repeat(1, repeats, 1, 1)[:, : self.in_chans]

    def _build_feature_stack(self, shot_gathers):
        torch = _require_torch()
        raw = self._normalize(shot_gathers)
        envelope = self._normalize(compute_envelope(shot_gathers, dim=-2))
        spectrogram = self._normalize(
            compute_spectrogram_energy(
                shot_gathers,
                n_fft=self.spectrogram_n_fft,
                hop_length=self.spectrogram_hop_length,
                win_length=self.spectrogram_win_length,
                power=self.spectrogram_power,
            )
        )
        raw_resized = resize_feature(raw, self.image_size)
        envelope_resized = resize_feature(envelope, self.image_size)
        spectrogram_resized = resize_feature(spectrogram, self.image_size)
        if self.feature_mode == "raw_repeat3":
            stacked = torch.stack([raw_resized, raw_resized, raw_resized], dim=1)
        elif self.feature_mode == "raw_envelope_spectrogram":
            stacked = torch.stack([raw_resized, envelope_resized, spectrogram_resized], dim=1)
        elif self.feature_mode == "raw_envelope":
            stacked = torch.stack([raw_resized, envelope_resized, 0.5 * (raw_resized + envelope_resized)], dim=1)
        elif self.feature_mode == "raw_spectrogram":
            stacked = torch.stack([raw_resized, spectrogram_resized, 0.5 * (raw_resized + spectrogram_resized)], dim=1)
        elif self.feature_mode == "spectrogram_multiband":
            stacked = torch.stack([spectrogram_resized, raw_resized * spectrogram_resized, 0.5 * (raw_resized + spectrogram_resized)], dim=1)
        elif self.feature_mode == "raw_envelope_spectrum3":
            stacked = torch.stack([raw_resized, envelope_resized, spectrogram_resized], dim=1)
        elif self.feature_mode == "envelope_repeat3":
            stacked = torch.stack([envelope_resized, envelope_resized, envelope_resized], dim=1)
        elif self.feature_mode == "spectrogram_repeat3":
            stacked = torch.stack([spectrogram_resized, spectrogram_resized, spectrogram_resized], dim=1)
        else:
            raise ValueError(f"unsupported feature_mode: {self.feature_mode}")
        stacked = torch.nan_to_num(stacked.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        return self._adapt_channels(stacked)

    def __call__(self, x):
        x = self._canonicalize(x)
        batch, shots, channels, time, receivers = x.shape
        grayscale = x.mean(dim=2).reshape(batch * shots, time, receivers)
        return self._build_feature_stack(grayscale)
