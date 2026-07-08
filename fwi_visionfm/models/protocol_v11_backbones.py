from __future__ import annotations

from typing import Any

from fwi_visionfm.models.foundation_fwi import FoundationModelFWI
from fwi_visionfm.models.protocol_v11_common_decoder import build_protocol_v11_decoder
from fwi_visionfm.models.seismic_bridge import SeismicToVisionBridge
from fwi_visionfm.torch_backend import require_torch_backend


class ProtocolV11CNNModel:
    def __init__(self, *, image_size: int, output_shape: tuple[int, int], vmin: float, vmax: float, base_channels: int = 16) -> None:
        torch = require_torch_backend()
        nn = torch.nn

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bridge = SeismicToVisionBridge(image_size=image_size, in_chans=3, norm_mode="zscore", feature_mode="raw_envelope_spectrum3")
                self.encoder = nn.Sequential(
                    nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GELU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
                    nn.AdaptiveAvgPool2d(1),
                )
                self.decoder = build_protocol_v11_decoder(output_shape=output_shape, base_channels=base_channels, vmin=vmin, vmax=vmax).module

            def forward(self, records, source_positions=None):
                batch, shots = records.shape[:2]
                images = self.bridge(records)
                feature = self.encoder(images).flatten(1).view(batch, shots, -1).mean(dim=1)
                return self.decoder(feature)

        self.module = _Module()

    def __getattr__(self, name: str) -> Any:
        if name == "module":
            raise AttributeError(name)
        return getattr(self.module, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)


class ProtocolV11VisionModel:
    def __init__(self, *, bridge: str, transfer_mode: str, pretrained: bool, config: dict[str, Any], vmin: float, vmax: float) -> None:
        torch = require_torch_backend()
        nn = torch.nn

        foundation = FoundationModelFWI(
            image_size=int(config["image_size"]),
            in_chans=3,
            backbone_type=str(config["backbones"]["dinov2_backbone_type"]),
            backbone_name=str(config["backbones"]["dinov2_model_name"]),
            pretrained=pretrained,
            transfer_mode=transfer_mode,
            bridge_norm="zscore",
            bridge_feature_mode=bridge,
            aggregation="mean",
            lora_rank=int(config["backbones"]["lora_rank"]),
            lora_alpha=float(config["backbones"]["lora_alpha"]),
            lora_dropout=float(config["backbones"]["lora_dropout"]),
            spectrogram_n_fft=int(config["bridges"]["spectrogram_multiband"]["n_fft"]),
            spectrogram_hop_length=int(config["bridges"]["spectrogram_multiband"]["hop_length"]),
            spectrogram_win_length=int(config["bridges"]["spectrogram_multiband"]["win_length"]),
            velocity_shape=tuple(config["velocity_shape"]),
            decoder_name="simple_bounded_decoder",
            decoder_hidden_dim=int(config["decoder_base_channels"]) * 16,
            vmin=vmin,
            vmax=vmax,
            print_parameter_report_flag=False,
        )

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.foundation = foundation
                self.decoder = foundation.head.module
                for parameter in self.foundation.aggregator.parameters():
                    parameter.requires_grad = True
                for parameter in self.decoder.parameters():
                    parameter.requires_grad = True

            def forward(self, records, source_positions=None):
                feature = self.foundation.extract_aggregated_features(records, source_positions)
                return self.decoder(feature)

        self.module = _Module()

    def __getattr__(self, name: str) -> Any:
        if name == "module":
            raise AttributeError(name)
        return getattr(self.module, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)


class ProtocolV11FeatureDecoderModel:
    def __init__(self, *, output_shape: tuple[int, int], vmin: float, vmax: float, base_channels: int = 16) -> None:
        torch = require_torch_backend()
        nn = torch.nn
        decoder = build_protocol_v11_decoder(output_shape=output_shape, base_channels=base_channels, vmin=vmin, vmax=vmax).module

        class _Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.decoder = decoder

            def forward(self, feature):
                return self.decoder(feature)

        self.module = _Module()

    def __getattr__(self, name: str) -> Any:
        if name == "module":
            raise AttributeError(name)
        return getattr(self.module, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)
