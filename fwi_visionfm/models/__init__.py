from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fwi_visionfm.bridge import bridge_multishot_record
from fwi_visionfm.config import BridgeConfig, ModelConfig

from .adapters import BottleneckAdapter, attach_adapters_to_vit
from .foundation_fwi import FWIModel, FoundationModelFWI
from .lora import LoRALinear, replace_linear_with_lora
from .parameter_utils import count_parameters, freeze_module, print_parameter_report, set_trainable_by_transfer_mode, unfreeze_module
from .seismic_bridge import SeismicToVisionBridge
from .vision_backbones import VisionBackboneWrapper, build_vision_backbone

Array = np.ndarray


def _sigmoid(x: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


class StatisticalBackbone:
    """Deterministic placeholder for future DINOv2/MAE/SAM encoders."""

    def encode(self, shot_image: Array) -> Array:
        image = np.asarray(shot_image, dtype=np.float32)
        per_channel_mean = np.mean(image, axis=(1, 2))
        per_channel_std = np.std(image, axis=(1, 2))
        global_stats = np.array(
            [np.min(image), np.max(image), np.mean(np.abs(image)), np.sqrt(np.mean(image * image))],
            dtype=np.float32,
        )
        return np.concatenate([per_channel_mean, per_channel_std, global_stats]).astype(np.float32)


def aggregate_features(features: Array, method: str = "mean", source_positions: Array | None = None) -> Array:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D (shots, feature_dim), got {features.shape}")
    if method == "mean":
        return np.mean(features, axis=0)
    if method == "attention":
        scores = np.mean(np.abs(features), axis=1)
        if source_positions is not None:
            centered = np.asarray(source_positions, dtype=np.float32) - 0.5
            scores = scores + 0.05 * (1.0 - np.abs(centered))
        weights = np.exp(scores - np.max(scores))
        weights = weights / (np.sum(weights) + 1.0e-6)
        return np.sum(features * weights[:, None], axis=0)
    if method == "source_attention":
        if source_positions is None:
            raise ValueError("source_attention aggregation requires source_positions")
        sources = np.asarray(source_positions, dtype=np.float32)
        if sources.shape != (features.shape[0],):
            raise ValueError(f"source_positions must have shape ({features.shape[0]},), got {sources.shape}")
        aperture = 1.0 - np.abs(sources - 0.5)
        energy = np.mean(np.abs(features), axis=1)
        scores = 0.75 * aperture + 0.25 * energy
        weights = np.exp(scores - np.max(scores))
        weights = weights / (np.sum(weights) + 1.0e-6)
        return np.sum(features * weights[:, None], axis=0)
    raise ValueError(f"unsupported aggregation method: {method}")


@dataclass
class VelocityDecoder:
    cfg: ModelConfig

    def decode(self, feature: Array) -> Array:
        feature = np.asarray(feature, dtype=np.float32)
        z = np.linspace(0.0, 1.0, self.cfg.velocity_depth, dtype=np.float32)[:, None]
        x = np.linspace(-1.0, 1.0, self.cfg.velocity_width, dtype=np.float32)[None, :]
        energy = float(np.mean(np.abs(feature)))
        contrast = float(np.std(feature))
        logits = -1.2 + 2.2 * z + 0.35 * np.sin(np.pi * x) + 0.25 * energy + 0.15 * contrast
        normalized = _sigmoid(logits)
        velocity = self.cfg.velocity_min + (self.cfg.velocity_max - self.cfg.velocity_min) * normalized
        return velocity.astype(np.float32)


class FWIVisionFMModel:
    def __init__(self, bridge: BridgeConfig | None = None, model: ModelConfig | None = None) -> None:
        self.bridge_cfg = bridge or BridgeConfig()
        self.model_cfg = model or ModelConfig()
        if self.model_cfg.backbone != "statistical":
            raise ValueError("only the statistical placeholder backbone is available in the NumPy scaffold")
        self.backbone = StatisticalBackbone()
        self.decoder = VelocityDecoder(self.model_cfg)

    def encode_records(self, records: Array) -> Array:
        bridged = bridge_multishot_record(records, self.bridge_cfg)
        return np.stack([self.backbone.encode(shot_image) for shot_image in bridged], axis=0)

    def predict(self, records: Array, source_positions: Array | None = None) -> Array:
        bridged = bridge_multishot_record(records, self.bridge_cfg, source_positions=source_positions)
        features = np.stack([self.backbone.encode(shot_image) for shot_image in bridged], axis=0)
        aggregate = aggregate_features(features, method=self.model_cfg.aggregation, source_positions=source_positions)
        return self.decoder.decode(aggregate)


__all__ = [
    "BottleneckAdapter",
    "FoundationModelFWI",
    "FWIModel",
    "FWIVisionFMModel",
    "LoRALinear",
    "SeismicToVisionBridge",
    "StatisticalBackbone",
    "VisionBackboneWrapper",
    "aggregate_features",
    "attach_adapters_to_vit",
    "build_vision_backbone",
    "count_parameters",
    "freeze_module",
    "print_parameter_report",
    "replace_linear_with_lora",
    "set_trainable_by_transfer_mode",
    "unfreeze_module",
]
