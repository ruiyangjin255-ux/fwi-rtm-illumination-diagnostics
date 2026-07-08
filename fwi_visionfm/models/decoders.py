from __future__ import annotations

import math
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def require_torch_backend():
    if torch is None:  # pragma: no cover
        raise RuntimeError("PyTorch backend is unavailable for decoder registry.")
    return torch


def _require_4d(x: Any) -> Any:
    torch = require_torch_backend()
    if x.ndim == 4:
        return x
    if x.ndim == 3:
        batch, tokens, channels = x.shape
        side = int(math.sqrt(tokens))
        if side * side == tokens:
            return x.transpose(1, 2).reshape(batch, channels, side, side)
        return x.mean(dim=1, keepdim=False)
    if x.ndim == 2:
        return x
    raise ValueError(f"decoder input must be 2-D, 3-D, or 4-D, got {tuple(x.shape)}")


class _SeedProjector:
    def __init__(self, base_channels: int, seed_shape: tuple[int, int]) -> None:
        torch = require_torch_backend()
        nn = torch.nn
        self.base_channels = int(base_channels)
        self.seed_shape = (int(seed_shape[0]), int(seed_shape[1]))
        self.vector_project = nn.LazyLinear(self.base_channels * self.seed_shape[0] * self.seed_shape[1])
        self.map_project = nn.LazyConv2d(self.base_channels, kernel_size=1)

    def __call__(self, x: Any) -> Any:
        torch = require_torch_backend()
        tensor = _require_4d(x)
        if tensor.ndim == 2:
            projected = self.vector_project(tensor)
            return projected.view(tensor.shape[0], self.base_channels, self.seed_shape[0], self.seed_shape[1])
        if tensor.ndim == 4:
            projected = self.map_project(tensor)
            return torch.nn.functional.interpolate(projected, size=self.seed_shape, mode="bilinear", align_corners=False)
        raise ValueError(f"unexpected tensor shape after adapter: {tuple(tensor.shape)}")


class _ConvBlock:
    def __init__(self, in_channels: int, out_channels: int) -> None:
        torch = require_torch_backend()
        nn = torch.nn
        self.module = nn.Sequential(
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(int(out_channels), int(out_channels), kernel_size=3, padding=1),
            nn.GELU(),
        )

    def __call__(self, x: Any) -> Any:
        return self.module(x)


class SimpleBoundedDecoder:
    def __init__(
        self,
        *,
        output_shape: tuple[int, int] = (70, 70),
        base_channels: int = 16,
        vmin: float = 1500.0,
        vmax: float = 4500.0,
        seed_shape: tuple[int, int] = (18, 18),
    ) -> None:
        torch = require_torch_backend()
        nn = torch.nn

        class _Module(nn.Module):
            def __init__(self, outer: "SimpleBoundedDecoder") -> None:
                super().__init__()
                self.outer = outer
                self.vector_project = outer.projector.vector_project
                self.map_project = outer.projector.map_project
                self.decoder = outer.decoder

            def forward(self, x: Any) -> Any:
                return self.outer._forward_impl(self, x)

        self.output_shape = (int(output_shape[0]), int(output_shape[1]))
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.projector = _SeedProjector(base_channels=base_channels, seed_shape=seed_shape)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(int(base_channels), int(base_channels), kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(int(base_channels), max(8, int(base_channels // 2)), kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(max(8, int(base_channels // 2)), 1, kernel_size=3, padding=1),
        )
        self.module = _Module(self)

    def _forward_impl(self, wrapper: Any, x: Any) -> Any:
        torch = require_torch_backend()
        seed = self.projector(x)
        logits = wrapper.decoder(seed)
        logits = torch.nn.functional.interpolate(logits, size=self.output_shape, mode="bilinear", align_corners=False)
        unit = torch.sigmoid(logits)
        return self.vmin + (self.vmax - self.vmin) * unit

    def __call__(self, x: Any) -> Any:
        return self.module(x)

    def to(self, device: str) -> "SimpleBoundedDecoder":
        self.module.to(device)
        return self

    def parameters(self):
        return self.module.parameters()


class UNetDecoder:
    def __init__(
        self,
        *,
        output_shape: tuple[int, int] = (70, 70),
        base_channels: int = 16,
        vmin: float = 1500.0,
        vmax: float = 4500.0,
        seed_shape: tuple[int, int] = (18, 18),
    ) -> None:
        torch = require_torch_backend()
        nn = torch.nn

        class _Module(nn.Module):
            def __init__(self, outer: "UNetDecoder") -> None:
                super().__init__()
                self.outer = outer
                self.vector_project = outer.projector.vector_project
                self.map_project = outer.projector.map_project
                self.enc1 = outer.enc1.module
                self.enc2 = outer.enc2.module
                self.bottleneck = outer.bottleneck.module
                self.up1 = outer.up1
                self.dec1 = outer.dec1.module
                self.up2 = outer.up2
                self.dec2 = outer.dec2.module
                self.head = outer.head

            def forward(self, x: Any) -> Any:
                return self.outer._forward_impl(self, x)

        self.output_shape = (int(output_shape[0]), int(output_shape[1]))
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.projector = _SeedProjector(base_channels=base_channels, seed_shape=seed_shape)
        self.enc1 = _ConvBlock(base_channels, base_channels)
        self.enc2 = _ConvBlock(base_channels, base_channels * 2)
        self.bottleneck = _ConvBlock(base_channels * 2, base_channels * 4)
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec1 = _ConvBlock(base_channels * 4, base_channels * 2)
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec2 = _ConvBlock(base_channels * 2, base_channels)
        self.head = nn.Conv2d(base_channels, 1, kernel_size=1)
        self.module = _Module(self)

    def _forward_impl(self, wrapper: Any, x: Any) -> Any:
        torch = require_torch_backend()
        seed = self.projector(x)
        enc1 = wrapper.enc1(seed)
        pooled1 = torch.nn.functional.max_pool2d(enc1, kernel_size=2)
        enc2 = wrapper.enc2(pooled1)
        pooled2 = torch.nn.functional.max_pool2d(enc2, kernel_size=2, ceil_mode=True)
        bottleneck = wrapper.bottleneck(pooled2)
        up1 = wrapper.up1(bottleneck)
        up1 = torch.nn.functional.interpolate(up1, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = wrapper.dec1(torch.cat([up1, enc2], dim=1))
        up2 = wrapper.up2(dec1)
        up2 = torch.nn.functional.interpolate(up2, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = wrapper.dec2(torch.cat([up2, enc1], dim=1))
        logits = wrapper.head(dec2)
        logits = torch.nn.functional.interpolate(logits, size=self.output_shape, mode="bilinear", align_corners=False)
        unit = torch.sigmoid(logits)
        return self.vmin + (self.vmax - self.vmin) * unit

    def __call__(self, x: Any) -> Any:
        return self.module(x)

    def to(self, device: str) -> "UNetDecoder":
        self.module.to(device)
        return self

    def parameters(self):
        return self.module.parameters()


def build_decoder(name: str, **kwargs: Any) -> Any:
    decoder_name = str(name)
    if decoder_name == "simple_bounded_decoder":
        return SimpleBoundedDecoder(**kwargs)
    if decoder_name == "unet_decoder":
        return UNetDecoder(**kwargs)
    if decoder_name == "boundary_aux_unet":
        from fwi_visionfm.models.boundary_aux_decoder import BoundaryAuxVelocityDecoder

        return BoundaryAuxVelocityDecoder(**kwargs)
    if decoder_name == "fpn_decoder":
        return SimpleBoundedDecoder(**kwargs)
    raise ValueError(f"unsupported decoder: {decoder_name}")
