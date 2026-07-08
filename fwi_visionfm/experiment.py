from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from fwi_visionfm.config import BridgeConfig, ModelConfig
from fwi_visionfm.train import run_npz_experiment


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    data_dir: str
    output_root: str = "fwi_visionfm/outputs/experiments"
    depth: int = 70
    width: int = 70
    channels: tuple[str, ...] = ("raw", "envelope")
    aggregation: str = "mean"
    batch_size: int = 2
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    seed: int = 0
    fit_linear_calibration: bool = False
    train_linear_epochs: int = 0
    linear_learning_rate: float = 1.0e-8

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["channels"] = list(self.channels)
        return data


def _config_from_dict(data: dict[str, Any]) -> ExperimentConfig:
    payload = dict(data)
    if "channels" in payload:
        payload["channels"] = tuple(payload["channels"])
    return ExperimentConfig(**payload)


def save_experiment_config(config: ExperimentConfig, path: str | Path) -> None:
    Path(path).write_text(json.dumps(config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return _config_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _safe_run_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    return cleaned or "experiment"


def _write_status(run_dir: Path, status: str, message: str, extra: dict[str, Any] | None = None) -> None:
    payload: dict[str, Any] = {
        "状态": status,
        "消息": message,
        "时间": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        payload.update(extra)
    (run_dir / "status.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_experiment_from_config(config: ExperimentConfig) -> dict[str, Any]:
    run_dir = Path(config.output_root) / _safe_run_name(config.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_experiment_config(config, run_dir / "resolved_config.json")
    _write_status(run_dir, "running", "实验已启动")
    try:
        summary = run_npz_experiment(
            data_dir=config.data_dir,
            output_dir=run_dir,
            bridge=BridgeConfig(channels=config.channels),
            model_config=ModelConfig(
                velocity_depth=config.depth,
                velocity_width=config.width,
                aggregation=config.aggregation,
            ),
            batch_size=config.batch_size,
            train_fraction=config.train_fraction,
            val_fraction=config.val_fraction,
            seed=config.seed,
            fit_linear_calibration=config.fit_linear_calibration,
            train_linear_epochs=config.train_linear_epochs,
            linear_learning_rate=config.linear_learning_rate,
        )
    except Exception as exc:
        _write_status(run_dir, "failed", str(exc))
        raise
    result = {
        "状态": "completed",
        "运行目录": str(run_dir),
        "摘要文件": str(run_dir / "npz_experiment_summary.json"),
        "样本数": summary["数据集"]["样本数"],
    }
    _write_status(run_dir, "completed", "实验完成", result)
    return result
