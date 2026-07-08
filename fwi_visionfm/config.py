from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    num_shots: int = 5
    num_receivers: int = 70
    num_time_samples: int = 200
    velocity_depth: int = 70
    velocity_width: int = 70
    noise_std: float = 0.0


@dataclass(frozen=True)
class BridgeConfig:
    channels: tuple[str, ...] = ("raw", "envelope")
    normalize: str = "maxabs"
    spectrum_bins: int = 16


@dataclass(frozen=True)
class ModelConfig:
    backbone: str = "statistical"
    aggregation: str = "mean"
    feature_dim: int = 12
    velocity_depth: int = 70
    velocity_width: int = 70
    velocity_min: float = 1400.0
    velocity_max: float = 4600.0


@dataclass(frozen=True)
class LossConfig:
    mae_weight: float = 1.0
    gradient_weight: float = 0.1
    ssim_weight: float = 0.1


@dataclass(frozen=True)
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    samples: int = 4
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("读取 YAML 配置需要 PyYAML，请先执行: python -m pip install pyyaml") from exc
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML 配置顶层必须是 mapping: {config_path}")
    return payload


def print_yaml_config(path: str | Path) -> str:
    payload = load_yaml_config(path)
    return str(payload)


def validate_config_paths(payload: dict[str, Any]) -> dict[str, list[str]]:
    existing_paths: list[str] = []
    missing_paths: list[str] = []

    def _walk(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                _walk(child)
            return
        if isinstance(value, list):
            for child in value:
                _walk(child)
            return
        if not isinstance(value, str):
            return
        normalized = value.replace("\\", "/").lower()
        if ":/" not in normalized and not normalized.startswith("/"):
            return
        candidate = Path(value)
        if candidate.exists():
            existing_paths.append(str(candidate))
        else:
            missing_paths.append(str(candidate))

    _walk(payload)
    return {
        "existing_paths": sorted(set(existing_paths)),
        "missing_paths": sorted(set(missing_paths)),
    }
