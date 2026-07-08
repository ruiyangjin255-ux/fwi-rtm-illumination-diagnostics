from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.calibration import (
    apply_linear_calibration,
    fit_linear_calibration as fit_linear_calibration_params,
    train_linear_calibration,
)
from fwi_visionfm.config import BridgeConfig, ModelConfig, RunConfig
from fwi_visionfm.datasets import NPZSampleDataset, discover_npz_samples, make_synthetic_sample, split_sample_paths
from fwi_visionfm.losses import combined_velocity_loss
from fwi_visionfm.metrics import velocity_metrics
from fwi_visionfm.models import FWIVisionFMModel


def _mean_dict(items: list[dict[str, Any]]) -> dict[str, Any]:
    keys = items[0].keys()
    result: dict[str, Any] = {}
    for key in keys:
        values = [item[key] for item in items if item.get(key) is not None]
        result[key] = None if not values else float(np.mean(values))
    return result


def run_smoke_experiment(output_dir: str | Path, cfg: RunConfig | None = None) -> dict[str, Any]:
    cfg = cfg or RunConfig()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model = FWIVisionFMModel(bridge=cfg.bridge, model=cfg.model)
    losses: list[dict[str, float]] = []
    metrics: list[dict[str, float]] = []
    last_prediction = None
    last_target = None
    for index in range(cfg.samples):
        sample = make_synthetic_sample(cfg.data, seed=cfg.seed + index)
        prediction = model.predict(sample.records, source_positions=sample.source_positions)
        losses.append(combined_velocity_loss(prediction, sample.velocity, cfg.loss))
        metrics.append(velocity_metrics(prediction, sample.velocity))
        last_prediction = prediction
        last_target = sample.velocity

    assert last_prediction is not None
    assert last_target is not None
    summary: dict[str, Any] = {
        "samples": int(cfg.samples),
        "config": cfg.to_dict(),
        "loss": _mean_dict(losses),
        "metrics": _mean_dict(metrics),
    }
    np.save(output_path / "prediction.npy", last_prediction.astype(np.float32))
    np.save(output_path / "target.npy", last_target.astype(np.float32))
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _evaluate_dataset(
    paths: list[Path],
    *,
    bridge: BridgeConfig,
    model_config: ModelConfig,
    batch_size: int,
) -> dict[str, Any]:
    if not paths:
        return {"样本数": 0, "batch数": 0, "loss": {}, "metrics": {}}
    dataset = NPZSampleDataset(paths)
    model = FWIVisionFMModel(bridge=bridge, model=model_config)
    losses: list[dict[str, float]] = []
    metrics: list[dict[str, float]] = []
    batch_count = 0
    for batch in dataset.iter_batches(batch_size):
        batch_count += 1
        for index in range(batch.records.shape[0]):
            prediction = model.predict(batch.records[index], source_positions=batch.source_positions[index])
            losses.append(combined_velocity_loss(prediction, batch.velocity[index]))
            metrics.append(velocity_metrics(prediction, batch.velocity[index]))
    return {
        "样本数": len(dataset),
        "batch数": batch_count,
        "loss": _mean_dict(losses),
        "metrics": _mean_dict(metrics),
    }


def _collect_predictions(
    paths: list[Path],
    *,
    bridge: BridgeConfig,
    model_config: ModelConfig,
    batch_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if not paths:
        return [], []
    dataset = NPZSampleDataset(paths)
    model = FWIVisionFMModel(bridge=bridge, model=model_config)
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for batch in dataset.iter_batches(batch_size):
        for index in range(batch.records.shape[0]):
            predictions.append(model.predict(batch.records[index], source_positions=batch.source_positions[index]))
            targets.append(batch.velocity[index])
    return predictions, targets


def _evaluate_dataset_with_calibration(
    paths: list[Path],
    *,
    bridge: BridgeConfig,
    model_config: ModelConfig,
    batch_size: int,
    calibration: dict[str, float],
) -> dict[str, Any]:
    predictions, targets = _collect_predictions(paths, bridge=bridge, model_config=model_config, batch_size=batch_size)
    if not predictions:
        return {"样本数": 0, "batch数": 0, "loss": {}, "metrics": {}}
    calibrated = [apply_linear_calibration(prediction, calibration) for prediction in predictions]
    losses = [combined_velocity_loss(prediction, target) for prediction, target in zip(calibrated, targets)]
    metrics = [velocity_metrics(prediction, target) for prediction, target in zip(calibrated, targets)]
    batch_count = int(np.ceil(len(predictions) / batch_size))
    return {
        "样本数": len(predictions),
        "batch数": batch_count,
        "loss": _mean_dict(losses),
        "metrics": _mean_dict(metrics),
    }


def run_npz_experiment(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    bridge: BridgeConfig | None = None,
    model_config: ModelConfig | None = None,
    batch_size: int = 2,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 0,
    fit_linear_calibration: bool = False,
    train_linear_epochs: int = 0,
    linear_learning_rate: float = 1.0e-8,
) -> dict[str, Any]:
    bridge = bridge or BridgeConfig()
    model_config = model_config or ModelConfig()
    paths = discover_npz_samples(data_dir)
    if not paths:
        raise ValueError(f"no npz samples found in {data_dir}")
    split = split_sample_paths(paths, train_fraction=train_fraction, val_fraction=val_fraction, seed=seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    split_summary = {
        name: _evaluate_dataset(
            split_paths,
            bridge=bridge,
            model_config=model_config,
            batch_size=batch_size,
        )
        for name, split_paths in split.items()
    }
    summary: dict[str, Any] = {
        "数据集": {
            "路径": str(Path(data_dir)),
            "样本数": len(paths),
        },
        "配置": {
            "bridge": bridge.__dict__,
            "model": model_config.__dict__,
            "batch_size": int(batch_size),
            "train_fraction": float(train_fraction),
            "val_fraction": float(val_fraction),
            "seed": int(seed),
            "fit_linear_calibration": bool(fit_linear_calibration),
            "train_linear_epochs": int(train_linear_epochs),
            "linear_learning_rate": float(linear_learning_rate),
        },
        "划分": split_summary,
    }
    train_predictions: list[np.ndarray] = []
    train_targets: list[np.ndarray] = []
    if fit_linear_calibration or train_linear_epochs > 0:
        train_predictions, train_targets = _collect_predictions(
            split["train"],
            bridge=bridge,
            model_config=model_config,
            batch_size=batch_size,
        )
        if not train_predictions:
            raise ValueError("cannot fit or train linear calibration because train split is empty")
    if fit_linear_calibration:
        calibration = fit_linear_calibration_params(train_predictions, train_targets)
        summary["线性校准"] = calibration
        summary["校准后划分"] = {
            name: _evaluate_dataset_with_calibration(
                split_paths,
                bridge=bridge,
                model_config=model_config,
                batch_size=batch_size,
                calibration=calibration,
            )
            for name, split_paths in split.items()
        }
    if train_linear_epochs > 0:
        train_result = train_linear_calibration(
            train_predictions,
            train_targets,
            epochs=train_linear_epochs,
            learning_rate=linear_learning_rate,
        )
        train_params = train_result["params"]
        history = train_result["history"]
        summary["线性训练"] = {
            "params": train_params,
            "history": history,
        }
        summary["线性训练后划分"] = {
            name: _evaluate_dataset_with_calibration(
                split_paths,
                bridge=bridge,
                model_config=model_config,
                batch_size=batch_size,
                calibration=train_params,
            )
            for name, split_paths in split.items()
        }
        history_json = output_path / "training_history.json"
        history_csv = output_path / "training_history.csv"
        history_json.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        with history_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["epoch", "mse", "mae", "scale", "bias", "grad_scale", "grad_bias"],
            )
            writer.writeheader()
            writer.writerows(history)
    (output_path / "npz_experiment_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary
