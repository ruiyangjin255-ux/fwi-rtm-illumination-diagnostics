"""PASD-FWI models: bridge selection, shared shot encoding, and velocity decoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .bridge import BridgeOutput, HybridAttributeBridge


BridgeMode = Literal["raw", "hybrid"]
DecoderMode = Literal["plain", "decoupled"]
AggregatorMode = Literal["mean", "geometry_attention"]


@dataclass
class PASDOutput:
    """Model outputs kept explicit for loss calculation, diagnostics, and plotting."""

    velocity: Tensor
    background: Tensor
    residual: Tensor
    attention: Tensor
    bridge: BridgeOutput


class ConvNormAct(nn.Module):
    """Two-convolution residual-free building block used by CPU-friendly encoders/decoders."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class SharedShotEncoder(nn.Module):
    """Compact encoder applied identically to every shot to retain shot-wise information."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 16,
        latent_channels: int = 96,
        latent_size: Tuple[int, int] = (9, 9),
    ) -> None:
        super().__init__()
        self.stem = ConvNormAct(in_channels, base_channels, stride=2)
        self.stage1 = ConvNormAct(base_channels, base_channels * 2, stride=2)
        self.stage2 = ConvNormAct(base_channels * 2, base_channels * 4, stride=2)
        self.proj = nn.Sequential(
            nn.Conv2d(base_channels * 4, latent_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(latent_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(latent_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(self.stage2(self.stage1(self.stem(x))))


class ShotAggregator(nn.Module):
    """Mean pooling or geometry-conditioned attention over per-shot feature maps."""

    def __init__(
        self,
        channels: int,
        geometry_dim: int = 3,
        mode: AggregatorMode = "geometry_attention",
    ) -> None:
        super().__init__()
        if mode not in {"mean", "geometry_attention"}:
            raise ValueError("mode must be 'mean' or 'geometry_attention'.")
        self.mode = mode
        hidden = max(16, channels // 2)
        self.score = nn.Sequential(
            nn.Linear(channels + geometry_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: Tensor, geometry: Tensor) -> Tuple[Tensor, Tensor]:
        if features.ndim != 5:
            raise ValueError("features must have shape [B,S,C,H,W].")
        b, s, _, _, _ = features.shape
        if geometry.shape[:2] != (b, s):
            raise ValueError("geometry must align with the first two feature dimensions.")
        if self.mode == "mean":
            attention = torch.full((b, s), 1.0 / s, device=features.device, dtype=features.dtype)
            return features.mean(dim=1), attention

        shot_global = features.mean(dim=(-2, -1))
        score_input = torch.cat([shot_global, geometry], dim=-1)
        attention = torch.softmax(self.score(score_input).squeeze(-1), dim=1)
        fused = (features * attention[:, :, None, None, None]).sum(dim=1)
        return fused, attention


class _DecodeBranch(nn.Module):
    """Small dense-prediction branch that restores a latent map to the velocity-grid size."""

    def __init__(self, in_channels: int, branch_channels: int, output_size: Tuple[int, int]) -> None:
        super().__init__()
        reduced = max(8, branch_channels // 2)
        self.body = nn.Sequential(
            ConvNormAct(in_channels, branch_channels),
            ConvNormAct(branch_channels, branch_channels),
            ConvNormAct(branch_channels, reduced),
        )
        self.head = nn.Conv2d(reduced, 1, kernel_size=1)
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        return self.head(F.interpolate(self.body(x), size=self.output_size, mode="bilinear", align_corners=False))


class PlainVelocityDecoder(nn.Module):
    """Single-head decoder for B1/B2 controls that use only ordinary velocity L1 supervision."""

    def __init__(self, in_channels: int, output_size: Tuple[int, int], branch_channels: int = 64) -> None:
        super().__init__()
        self.branch = _DecodeBranch(in_channels, branch_channels, output_size)

    def forward(self, fused: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        velocity = torch.sigmoid(self.branch(fused))
        return velocity, velocity, torch.zeros_like(velocity)


class BackgroundEdgeDecoder(nn.Module):
    """Predict low-wavenumber background and bounded high-wavenumber residual separately."""

    def __init__(
        self,
        in_channels: int,
        output_size: Tuple[int, int] = (70, 70),
        branch_channels: int = 64,
        residual_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.background_branch = _DecodeBranch(in_channels, branch_channels, output_size)
        self.edge_branch = _DecodeBranch(in_channels, branch_channels, output_size)
        self.residual_scale = float(residual_scale)

    def forward(self, fused: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        background = torch.sigmoid(self.background_branch(fused))
        residual = self.residual_scale * torch.tanh(self.edge_branch(fused))
        velocity = (background + residual).clamp(0.0, 1.0)
        return velocity, background, residual


class PASDFWI(nn.Module):
    """End-to-end multi-shot FWI model for targets normalized to [0, 1].

    The B1--B4 ablations are not emulated by post-hoc masking: they use explicit bridge and
    decoder modes so the control models remain scientifically interpretable.
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (70, 70),
        base_channels: int = 16,
        latent_channels: int = 96,
        latent_size: Tuple[int, int] = (9, 9),
        aggregator: AggregatorMode = "geometry_attention",
        bridge_mode: BridgeMode = "hybrid",
        decoder_mode: DecoderMode = "decoupled",
        residual_scale: float = 0.25,
        bridge: HybridAttributeBridge | None = None,
    ) -> None:
        super().__init__()
        if bridge_mode not in {"raw", "hybrid"}:
            raise ValueError("bridge_mode must be 'raw' or 'hybrid'.")
        if decoder_mode not in {"plain", "decoupled"}:
            raise ValueError("decoder_mode must be 'plain' or 'decoupled'.")
        self.bridge_mode = bridge_mode
        self.decoder_mode = decoder_mode
        self.bridge = bridge if bridge is not None else HybridAttributeBridge()
        in_channels = 1 if bridge_mode == "raw" else 3
        self.encoder = SharedShotEncoder(
            in_channels=in_channels,
            base_channels=base_channels,
            latent_channels=latent_channels,
            latent_size=latent_size,
        )
        self.aggregator = ShotAggregator(latent_channels, mode=aggregator)
        branch_channels = max(32, latent_channels // 2)
        if decoder_mode == "plain":
            self.decoder: nn.Module = PlainVelocityDecoder(latent_channels, output_size, branch_channels)
        else:
            self.decoder = BackgroundEdgeDecoder(
                in_channels=latent_channels,
                output_size=output_size,
                branch_channels=branch_channels,
                residual_scale=residual_scale,
            )

    def _select_attributes(self, attributes: Tensor) -> Tensor:
        if self.bridge_mode == "raw":
            return attributes[:, :, :1]
        return attributes

    def forward(
        self,
        records: Tensor,
        source_positions: Tensor | None = None,
        receiver_positions: Tensor | None = None,
    ) -> PASDOutput:
        bridge_output = self.bridge(records, source_positions, receiver_positions)
        attributes = self._select_attributes(bridge_output.attributes)
        b, s, c, t, r = attributes.shape
        encoded = self.encoder(attributes.reshape(b * s, c, t, r))
        _, channels, h, w = encoded.shape
        encoded = encoded.reshape(b, s, channels, h, w)
        fused, attention = self.aggregator(encoded, bridge_output.geometry)
        velocity, background, residual = self.decoder(fused)  # type: ignore[misc]
        return PASDOutput(
            velocity=velocity,
            background=background,
            residual=residual,
            attention=attention,
            bridge=bridge_output,
        )
