from __future__ import annotations

from typing import Any

from fwi_visionfm.models.adapters import attach_adapters_to_vit
from fwi_visionfm.models.decoders import build_decoder
from fwi_visionfm.models.lora import replace_linear_with_lora
from fwi_visionfm.models.parameter_utils import print_parameter_report, set_trainable_by_transfer_mode
from fwi_visionfm.models.seismic_bridge import SeismicToVisionBridge
from fwi_visionfm.models.vision_backbones import build_vision_backbone
from fwi_visionfm.optional_deps import missing_dependencies

if not missing_dependencies("torch"):
    import torch

    _FoundationBase = torch.nn.Module
else:
    _FoundationBase = object


def _require_torch():
    if missing_dependencies("torch"):
        raise RuntimeError(
            "PyTorch backend is unavailable. Install PyTorch first, then rerun this experiment. "
            "Suggested CPU install: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
        )
    import torch

    return torch


class CrossShotAggregator(_FoundationBase):
    def __init__(self, feature_dim: int, aggregation: str = "mean") -> None:
        torch = _require_torch()
        nn = torch.nn
        super().__init__()
        self.aggregation = str(aggregation)
        if self.aggregation not in {"mean", "attention", "source_attention", "gated_mean"}:
            raise ValueError(f"unsupported aggregation: {aggregation}")
        self.attn = nn.Linear(feature_dim, 1) if self.aggregation in {"attention", "source_attention"} else None
        self.gate = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.Sigmoid()) if self.aggregation == "gated_mean" else None

    def forward(self, features, source_positions=None):
        torch = _require_torch()
        if self.aggregation == "mean":
            return features.mean(dim=1)
        if self.aggregation == "gated_mean":
            gate = self.gate(features)
            return (features * gate).mean(dim=1)
        scores = self.attn(features).squeeze(-1)
        if self.aggregation == "source_attention" and source_positions is not None:
            scores = scores + 0.1 * (1.0 - (source_positions.to(features.device) - 0.5).abs())
        weights = torch.softmax(scores, dim=1)
        return (features * weights.unsqueeze(-1)).sum(dim=1)


class VelocityRegressionHead(_FoundationBase):
    def __init__(self, input_dim: int, velocity_shape: tuple[int, int], hidden_dim: int = 256) -> None:
        torch = _require_torch()
        nn = torch.nn
        super().__init__()
        self.velocity_shape = (int(velocity_shape[0]), int(velocity_shape[1]))
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.velocity_shape[0] * self.velocity_shape[1]),
        )

    def forward(self, x):
        return self.net(x).view(x.shape[0], *self.velocity_shape)


class FoundationModelFWI(_FoundationBase):
    def __init__(
        self,
        image_size: int = 224,
        in_chans: int = 3,
        backbone_type: str = "dummy",
        backbone_name: str = "vit_tiny_patch16_224",
        pretrained: bool = False,
        transfer_mode: str = "frozen",
        remove_cls_token: bool = False,
        bridge_norm: str = "zscore",
        bridge_feature_mode: str = "raw_repeat3",
        aggregation: str = "mean",
        adapter_bottleneck: int = 64,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        spectrogram_n_fft: int = 64,
        spectrogram_hop_length: int = 16,
        spectrogram_win_length: int = 64,
        spectrogram_power: float = 1.0,
        velocity_shape: tuple[int, int] = (70, 70),
        decoder_hidden_dim: int = 256,
        decoder_name: str = "simple_bounded_decoder",
        decoder_kwargs: dict[str, Any] | None = None,
        vmin: float = 1500.0,
        vmax: float = 4500.0,
        local_files_only: bool = False,
        print_parameter_report_flag: bool = True,
    ) -> None:
        torch = _require_torch()
        super().__init__()
        self.image_size = int(image_size)
        self.in_chans = int(in_chans)
        self.backbone_type = str(backbone_type)
        self.backbone_name = str(backbone_name)
        self.pretrained = bool(pretrained)
        self.transfer_mode = str(transfer_mode)
        self.remove_cls_token = bool(remove_cls_token)
        self.bridge_norm = str(bridge_norm)
        self.bridge_feature_mode = str(bridge_feature_mode)
        self.aggregation_name = str(aggregation)
        self.velocity_shape = (int(velocity_shape[0]), int(velocity_shape[1]))
        self.local_files_only = bool(local_files_only)
        self.bridge = SeismicToVisionBridge(
            image_size=self.image_size,
            in_chans=self.in_chans,
            norm_mode=self.bridge_norm,
            feature_mode=self.bridge_feature_mode,
            spectrogram_n_fft=int(spectrogram_n_fft),
            spectrogram_hop_length=int(spectrogram_hop_length),
            spectrogram_win_length=int(spectrogram_win_length),
            spectrogram_power=float(spectrogram_power),
        )
        effective_pretrained = False if self.transfer_mode == "scratch" else self.pretrained
        self.backbone = build_vision_backbone(
            backbone_type=self.backbone_type,
            model_name=self.backbone_name,
            pretrained=effective_pretrained,
            image_size=self.image_size,
            in_chans=self.in_chans,
            freeze=False,
            remove_cls_token=self.remove_cls_token,
            local_files_only=self.local_files_only,
        )
        self.injected_adapter_modules = 0
        self.injected_lora_modules = 0
        if self.transfer_mode == "adapter":
            self.injected_adapter_modules = attach_adapters_to_vit(
                self.backbone.backbone,
                bottleneck_dim=int(adapter_bottleneck),
                dropout=0.0,
                scale=1.0,
            )
        elif self.transfer_mode == "lora":
            self.injected_lora_modules = replace_linear_with_lora(
                self.backbone.backbone,
                r=int(lora_rank),
                alpha=float(lora_alpha),
                dropout=float(lora_dropout),
                freeze_base=True,
            )
        self.aggregator = CrossShotAggregator(feature_dim=self.backbone.embed_dim, aggregation=self.aggregation_name)
        self.decoder_name = str(decoder_name)
        self.decoder_kwargs = dict(decoder_kwargs or {})
        self.head = build_decoder(
            self.decoder_name,
            output_shape=self.velocity_shape,
            base_channels=max(16, int(decoder_hidden_dim // 16)),
            vmin=float(vmin),
            vmax=float(vmax),
            **self.decoder_kwargs,
        )
        set_trainable_by_transfer_mode(self, self.transfer_mode)
        self.parameter_report = (
            print_parameter_report(
                self,
                title="FoundationModelFWI",
                metadata={
                    "transfer_mode": self.transfer_mode,
                    "backbone_type": self.backbone_type,
                    "backbone_name": self.backbone_name,
                    "bridge_feature_mode": self.bridge_feature_mode,
                    "decoder_name": self.decoder_name,
                },
            )
            if print_parameter_report_flag
            else None
        )

    def extract_aggregated_features(self, x, source_positions=None):
        if x.ndim not in {4, 5}:
            raise ValueError(f"expected input shape [B,S,T,R] or [B,S,C,T,R], got {tuple(x.shape)}")
        batch = x.shape[0]
        shots = x.shape[1]
        images = self.bridge(x)
        tokens = self.backbone(images)
        if tokens.ndim != 3:
            raise RuntimeError(f"backbone must return [B, N, D] tokens, got {tuple(tokens.shape)}")
        pooled = tokens.mean(dim=1)
        shot_features = pooled.view(batch, shots, self.backbone.embed_dim)
        if source_positions is None:
            torch = _require_torch()
            source_positions = torch.linspace(0.0, 1.0, shots, dtype=shot_features.dtype, device=shot_features.device)
            source_positions = source_positions.unsqueeze(0).expand(batch, shots)
        return self.aggregator(shot_features, source_positions=source_positions)

    def forward(self, x, source_positions=None):
        aggregated = self.extract_aggregated_features(x, source_positions=source_positions)
        return self.head(aggregated)[:, 0]


FWIModel = FoundationModelFWI
