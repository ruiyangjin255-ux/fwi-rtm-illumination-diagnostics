from __future__ import annotations

from typing import Any

from fwi_visionfm.models.decoders import UNetDecoder, require_torch_backend


class BoundaryAuxVelocityDecoder:
    def __init__(
        self,
        *,
        output_shape: tuple[int, int] = (70, 70),
        base_channels: int = 16,
        vmin: float = 1500.0,
        vmax: float = 4500.0,
        seed_shape: tuple[int, int] = (18, 18),
        return_boundary: bool = True,
    ) -> None:
        torch = require_torch_backend()
        nn = torch.nn

        class _Module(nn.Module):
            def __init__(self, outer: "BoundaryAuxVelocityDecoder") -> None:
                super().__init__()
                self.outer = outer
                self.vector_project = outer.base_decoder.projector.vector_project
                self.map_project = outer.base_decoder.projector.map_project
                self.enc1 = outer.base_decoder.enc1.module
                self.enc2 = outer.base_decoder.enc2.module
                self.bottleneck = outer.base_decoder.bottleneck.module
                self.up1 = outer.base_decoder.up1
                self.dec1 = outer.base_decoder.dec1.module
                self.up2 = outer.base_decoder.up2
                self.dec2 = outer.base_decoder.dec2.module
                self.velocity_head = outer.velocity_head
                self.boundary_head = outer.boundary_head

            def forward(self, x: Any) -> Any:
                return self.outer._forward_impl(self, x)

        self.output_shape = (int(output_shape[0]), int(output_shape[1]))
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.return_boundary = bool(return_boundary)
        self.base_decoder = UNetDecoder(output_shape=output_shape, base_channels=base_channels, vmin=vmin, vmax=vmax, seed_shape=seed_shape)
        self.velocity_head = self.base_decoder.head
        self.boundary_head = nn.Sequential(
            nn.Conv2d(base_channels, max(4, base_channels // 2), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(max(4, base_channels // 2), 1, kernel_size=1),
        )
        self.module = _Module(self)

    def _forward_impl(self, wrapper: Any, x: Any) -> Any:
        torch = require_torch_backend()
        seed = self.base_decoder.projector(x)
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
        velocity_logits = wrapper.velocity_head(dec2)
        velocity_logits = torch.nn.functional.interpolate(velocity_logits, size=self.output_shape, mode="bilinear", align_corners=False)
        velocity = self.vmin + (self.vmax - self.vmin) * torch.sigmoid(velocity_logits)
        if not self.return_boundary:
            return {"velocity": velocity}
        boundary = torch.sigmoid(wrapper.boundary_head(dec2))
        boundary = torch.nn.functional.interpolate(boundary, size=self.output_shape, mode="bilinear", align_corners=False)
        return {"velocity": velocity, "boundary": boundary}

    def __call__(self, x: Any) -> Any:
        return self.module(x)

    def to(self, device: str) -> "BoundaryAuxVelocityDecoder":
        self.module.to(device)
        return self

    def parameters(self):
        return self.module.parameters()
