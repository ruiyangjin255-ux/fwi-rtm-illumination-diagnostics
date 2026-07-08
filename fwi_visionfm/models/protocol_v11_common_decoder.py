from __future__ import annotations

from typing import Any

from fwi_visionfm.models.decoders import build_decoder


class CommonBoundedVelocityDecoder:
    """Shared decoder contract for every Protocol V11 backbone."""

    def __init__(
        self,
        *,
        output_shape: tuple[int, int] = (70, 70),
        base_channels: int = 16,
        vmin: float = 1500.0,
        vmax: float = 4500.0,
    ) -> None:
        self.decoder = build_decoder(
            "simple_bounded_decoder",
            output_shape=output_shape,
            base_channels=base_channels,
            vmin=vmin,
            vmax=vmax,
        )
        self.module = self.decoder.module

    def __call__(self, feature: Any) -> Any:
        return self.module(feature)


def build_protocol_v11_decoder(**kwargs: Any) -> CommonBoundedVelocityDecoder:
    return CommonBoundedVelocityDecoder(**kwargs)

