from __future__ import annotations

from typing import Any

import numpy as np

from fwi_visionfm.models.decoders import build_decoder
from fwi_visionfm.models import FoundationModelFWI, SeismicToVisionBridge
from fwi_visionfm.peft import (
    AdapterConfig,
    LoRAConfig,
)
from fwi_visionfm.data.bridge_registry import build_bridge
from fwi_visionfm.torch_backend import require_torch_backend


def _normalize_raw(records):
    scale = records.abs().amax(dim=(-1, -2), keepdim=True).clamp_min(1.0e-6)
    return records / scale


def _zscore_raw(records):
    mean = records.mean(dim=(-1, -2), keepdim=True)
    std = records.std(dim=(-1, -2), keepdim=True).clamp_min(1.0e-6)
    return (records - mean) / std


def build_torch_shot_images(records, source_positions, *, channels: tuple[str, ...], bridge_mode: str = "simple"):
    torch = require_torch_backend()
    if records.ndim != 4:
        raise ValueError(f"records must have shape (batch, shots, receivers, time), got {tuple(records.shape)}")
    batch, shots, receivers, time = records.shape
    if tuple(source_positions.shape) != (batch, shots):
        raise ValueError(f"source_positions must have shape {(batch, shots)}, got {tuple(source_positions.shape)}")
    if bridge_mode == "simple":
        normalized_records = _normalize_raw(records)
    elif bridge_mode == "normalized":
        normalized_records = _zscore_raw(records)
    elif bridge_mode == "channel_stack":
        normalized_records = _normalize_raw(records)
    else:
        raise ValueError(f"unsupported bridge_mode: {bridge_mode}")
    channel_tensors = []
    for name in channels:
        if name == "raw":
            channel_tensors.append(normalized_records)
        elif name == "offset":
            receiver_positions = torch.linspace(0.0, 1.0, receivers, dtype=records.dtype, device=records.device)
            offsets = receiver_positions.view(1, 1, receivers, 1) - source_positions.to(records.device).view(batch, shots, 1, 1)
            channel_tensors.append(offsets.expand(batch, shots, receivers, time))
        else:
            raise ValueError(f"unsupported torch bridge channel: {name}")
    images = torch.stack(channel_tensors, dim=2)
    if bridge_mode == "channel_stack":
        stacked = normalized_records.mean(dim=1, keepdim=True).expand(batch, shots, receivers, time)
        images = torch.cat([images, stacked.unsqueeze(2)], dim=2)
    return images


def build_pseudo_vision_images(records, source_positions, *, image_size: int = 224):
    torch = require_torch_backend()
    if records.ndim != 4:
        raise ValueError(f"records must have shape (batch, shots, receivers, time), got {tuple(records.shape)}")
    batch, shots, receivers, time = records.shape
    if tuple(source_positions.shape) != (batch, shots):
        raise ValueError(f"source_positions must have shape {(batch, shots)}, got {tuple(source_positions.shape)}")
    raw = _normalize_raw(records)
    amplitude = raw.abs()
    receiver_positions = torch.linspace(0.0, 1.0, receivers, dtype=records.dtype, device=records.device)
    offsets = receiver_positions.view(1, 1, receivers, 1) - source_positions.to(records.device).view(batch, shots, 1, 1)
    stacked = torch.stack([raw, amplitude, offsets.expand(batch, shots, receivers, time)], dim=2)
    flat = stacked.view(batch * shots, 3, receivers, time)
    resized = torch.nn.functional.interpolate(
        flat,
        size=(int(image_size), int(image_size)),
        mode="bilinear",
        align_corners=False,
    )
    return resized.view(batch, shots, 3, int(image_size), int(image_size))


class SeismicToVisionTorchBridge:
    def __init__(self, channels: tuple[str, ...] = ("raw",), bridge_mode: str = "simple") -> None:
        self.channels = tuple(channels)
        self.bridge_mode = str(bridge_mode)

    def __call__(self, records, source_positions):
        return build_torch_shot_images(records, source_positions, channels=self.channels, bridge_mode=self.bridge_mode)


class PseudoVisionImageBridge:
    def __init__(self, image_size: int = 224) -> None:
        self.image_size = int(image_size)

    def __call__(self, records, source_positions):
        return build_pseudo_vision_images(records, source_positions, image_size=self.image_size)


class ShotEncoderCNN:
    def __init__(self, in_channels: int, feature_dim: int = 64) -> None:
        torch = require_torch_backend()
        nn = torch.nn
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, feature_dim),
            nn.GELU(),
        )


class SimpleVisionBackbone(ShotEncoderCNN):
    pass


class CrossShotAggregator:
    def __init__(self, aggregation: str, feature_dim: int) -> None:
        torch = require_torch_backend()
        nn = torch.nn
        self.aggregation = aggregation
        self.attn = nn.Linear(feature_dim + 1, 1) if aggregation == "source_attention" else None

    def __call__(self, features, source_positions):
        torch = require_torch_backend()
        if self.aggregation == "mean":
            return features.mean(dim=1)
        if self.aggregation == "max":
            return features.max(dim=1).values
        if self.aggregation == "source_attention":
            sources = source_positions.to(features.device).unsqueeze(-1)
            scores = self.attn(torch.cat([features, sources], dim=-1)).squeeze(-1)
            weights = torch.softmax(scores, dim=1)
            return (features * weights.unsqueeze(-1)).sum(dim=1)
        raise ValueError(f"unsupported aggregation: {self.aggregation}")


class VelocityDecoderCNN:
    def __init__(self, feature_dim: int, depth: int, width: int) -> None:
        torch = require_torch_backend()
        nn = torch.nn
        self.depth = int(depth)
        self.width = int(width)
        hidden_h = max(4, int(np.ceil(depth / 4)))
        hidden_w = max(4, int(np.ceil(width / 4)))
        self.hidden_shape = (32, hidden_h, hidden_w)
        self.project = nn.Sequential(
            nn.Linear(feature_dim, int(np.prod(self.hidden_shape))),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )

    def __call__(self, features):
        torch = require_torch_backend()
        x = self.project(features).view(features.shape[0], *self.hidden_shape)
        logits = self.decoder(x)
        logits = torch.nn.functional.interpolate(logits, size=(self.depth, self.width), mode="bilinear", align_corners=False)
        return logits[:, 0]


class BoundedVelocityDecoder:
    def __init__(self, decoder: VelocityDecoderCNN, vmin: float = 1500.0, vmax: float = 4500.0, decoder_mode: str = "bounded") -> None:
        self.decoder = decoder
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.decoder_mode = str(decoder_mode)

    def __call__(self, features):
        logits = self.decoder(features)
        if self.decoder_mode == "unbounded":
            return logits
        torch = require_torch_backend()
        velocity_unit = torch.sigmoid(logits)
        return self.vmin + (self.vmax - self.vmin) * velocity_unit


class FwiVisionFmTorchBaseline:
    def __init__(
        self,
        *,
        channels: tuple[str, ...] = ("raw",),
        depth: int = 70,
        width: int = 70,
        aggregation: str = "mean",
        bridge_mode: str = "simple",
        decoder_mode: str = "bounded",
        feature_dim: int = 64,
        vmin: float = 1500.0,
        vmax: float = 4500.0,
        decoder_name: str = "simple_bounded_decoder",
        decoder_kwargs: dict[str, Any] | None = None,
        bridge_name: str | None = None,
        bridge_config: dict[str, Any] | None = None,
    ) -> None:
        torch = require_torch_backend()
        nn = torch.nn

        class _Model(nn.Module):
            def __init__(self, outer: "FwiVisionFmTorchBaseline") -> None:
                super().__init__()
                self.outer = outer
                self.encoder = outer.backbone.module
                self.attn = outer.aggregator.attn
                self.velocity_decoder = outer.velocity_decoder.module

            def forward(self, records, source_positions):
                return self.outer._forward_impl(self, records, source_positions)

        self.channels = tuple(channels)
        self.depth = int(depth)
        self.width = int(width)
        self.feature_dim = int(feature_dim)
        self.bridge_mode = str(bridge_mode)
        self.decoder_mode = str(decoder_mode)
        self.bridge_name = bridge_name
        self.registry_bridge = build_bridge(bridge_name, bridge_config or {}) if bridge_name else None
        self.bridge = SeismicToVisionTorchBridge(self.channels, bridge_mode=self.bridge_mode)
        encoder_channels = self.registry_bridge.output_channels if self.registry_bridge else len(self.channels) + (1 if self.bridge_mode == "channel_stack" else 0)
        self.backbone = SimpleVisionBackbone(encoder_channels, feature_dim=self.feature_dim)
        self.aggregator = CrossShotAggregator(aggregation=aggregation, feature_dim=self.feature_dim)
        self.decoder_name = str(decoder_name)
        self.decoder_kwargs = dict(decoder_kwargs or {})
        self.velocity_decoder = build_decoder(
            self.decoder_name,
            output_shape=(self.depth, self.width),
            vmin=vmin,
            vmax=vmax,
            **self.decoder_kwargs,
        )
        self.module = _Model(self)

    def parameters(self):
        return self.module.parameters()

    def train(self) -> None:
        self.module.train()

    def eval(self) -> None:
        self.module.eval()

    def to(self, device: str):
        self.module.to(device)
        return self

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        return self.module.load_state_dict(state_dict)

    def __call__(self, records, source_positions):
        return self.module(records, source_positions)

    def _forward_impl(self, wrapper, records, source_positions):
        if self.registry_bridge is not None:
            image = self.registry_bridge.forward(records)["image"]
            features = wrapper.encoder(image)
            return wrapper.velocity_decoder(features)[:, 0]
        images = self.bridge(records, source_positions)
        batch, shots, channels, receivers, time = images.shape
        flat = images.view(batch * shots, channels, receivers, time)
        features = wrapper.encoder(flat).view(batch, shots, self.feature_dim)
        aggregate = self.aggregator(features, source_positions)
        return wrapper.velocity_decoder(aggregate)[:, 0]


FWITorchBaseline = FwiVisionFmTorchBaseline


class FrozenFoundationFWI:
    def __init__(
        self,
        *,
        foundation_backbone: str = "dummy_dinov2",
        backbone_type: str | None = None,
        model_name: str | None = None,
        pretrained: bool = False,
        freeze_backbone: bool = True,
        peft_type: str = "none",
        lora_config: LoRAConfig | None = None,
        adapter_config: AdapterConfig | None = None,
        image_size: int = 224,
        in_chans: int = 3,
        norm_mode: str = "zscore",
        remove_cls_token: bool = False,
        local_files_only: bool = False,
        depth: int = 70,
        width: int = 70,
        aggregation: str = "mean",
        bridge_feature_mode: str = "raw_repeat3",
        spectrogram_n_fft: int = 64,
        spectrogram_hop_length: int = 16,
        spectrogram_win_length: int = 64,
        spectrogram_power: float = 1.0,
        vmin: float = 1500.0,
        vmax: float = 4500.0,
        device: str = "cpu",
        transfer_mode: str | None = None,
        print_parameter_report: bool = True,
        decoder_name: str = "simple_bounded_decoder",
        decoder_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.depth = int(depth)
        self.width = int(width)
        self.image_size = int(image_size)
        self.peft_type = peft_type
        resolved_backbone_type, resolved_model_name = self._resolve_backbone_config(
            foundation_backbone=foundation_backbone,
            backbone_type=backbone_type,
            model_name=model_name,
        )
        if transfer_mode is None:
            if peft_type == "adapter":
                transfer_mode = "adapter"
            elif peft_type == "lora":
                transfer_mode = "lora"
            elif freeze_backbone:
                transfer_mode = "frozen"
            else:
                transfer_mode = "full"
        adapter_bottleneck = int((adapter_config or AdapterConfig()).bottleneck_dim)
        lora_cfg = lora_config or LoRAConfig()
        self.module = FoundationModelFWI(
            image_size=image_size,
            in_chans=in_chans,
            backbone_type=resolved_backbone_type,
            backbone_name=resolved_model_name,
            pretrained=pretrained,
            transfer_mode=str(transfer_mode),
            remove_cls_token=remove_cls_token,
            bridge_norm=norm_mode,
            bridge_feature_mode=bridge_feature_mode,
            aggregation=aggregation,
            adapter_bottleneck=adapter_bottleneck,
            lora_rank=int(lora_cfg.rank),
            lora_alpha=float(lora_cfg.alpha),
            lora_dropout=float(lora_cfg.dropout),
            spectrogram_n_fft=int(spectrogram_n_fft),
            spectrogram_hop_length=int(spectrogram_hop_length),
            spectrogram_win_length=int(spectrogram_win_length),
            spectrogram_power=float(spectrogram_power),
            velocity_shape=(self.depth, self.width),
            decoder_hidden_dim=256,
            decoder_name=str(decoder_name),
            decoder_kwargs=dict(decoder_kwargs or {}),
            vmin=float(vmin),
            vmax=float(vmax),
            local_files_only=local_files_only,
            print_parameter_report_flag=print_parameter_report,
        ).to(device)
        self.injected_lora_modules = int(getattr(self.module, "injected_lora_modules", 0))
        self.injected_adapter_modules = int(getattr(self.module, "injected_adapter_modules", 0))
        self.foundation_encoder = self.module.backbone
        self.feature_dim = int(self.foundation_encoder.embed_dim)

    def parameters(self):
        return self.module.parameters()

    def train(self) -> None:
        self.module.train()

    def eval(self) -> None:
        self.module.eval()

    def to(self, device: str):
        self.module.to(device)
        return self

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        return self.module.load_state_dict(state_dict)

    def named_parameters(self):
        return self.module.named_parameters()

    def __call__(self, records, source_positions):
        return self.module(records, source_positions)

    @staticmethod
    def _resolve_backbone_config(*, foundation_backbone: str, backbone_type: str | None, model_name: str | None) -> tuple[str, str]:
        if backbone_type is not None:
            resolved_name = model_name or ("dummy_dinov2" if backbone_type == "dummy" else "vit_tiny_patch16_224")
            return str(backbone_type), str(resolved_name)
        if foundation_backbone == "dummy_dinov2":
            return "dummy", "dummy_dinov2"
        if str(foundation_backbone).startswith("facebook/"):
            return "hf_dinov2", str(model_name or foundation_backbone)
        return "timm", str(model_name or foundation_backbone)
