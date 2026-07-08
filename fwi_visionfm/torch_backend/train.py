from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.config import DataConfig
from fwi_visionfm.data.openfwi_npy_dataset import OpenFWINpyDataset
from fwi_visionfm.data_conversion import convert_array_dataset_to_npz
from fwi_visionfm.datasets import discover_npz_samples, make_synthetic_sample, split_sample_paths
from fwi_visionfm.metrics import velocity_metrics
from fwi_visionfm.torch_backend import require_torch_backend
from fwi_visionfm.torch_backend.data import build_torch_dataloader
from fwi_visionfm.torch_backend.model import FwiVisionFmTorchBaseline


def count_model_parameters(model: Any) -> dict[str, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        total += count
        if bool(getattr(parameter, "requires_grad", False)):
            trainable += count
    ratio = float(trainable / total) if total > 0 else 0.0
    return {"total_parameters": total, "trainable_parameters": trainable, "trainable_ratio": ratio}


def set_torch_seed(seed: int) -> None:
    torch = require_torch_backend()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def torch_metrics(prediction, target) -> dict[str, float]:
    torch = require_torch_backend()
    diff = prediction.detach() - target.detach()
    mae = diff.abs().mean()
    rmse = torch.sqrt((diff * diff).mean())
    return {"mae": float(mae.cpu()), "rmse": float(rmse.cpu())}


def save_checkpoint(output_dir: str | Path, epoch: int, model: FwiVisionFmTorchBaseline, optimizer: Any, metrics: dict[str, float]) -> Path:
    torch = require_torch_backend()
    path = Path(output_dir) / "checkpoints" / f"epoch_{epoch:03d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    return path


def save_last_checkpoint(output_dir: str | Path, model: FwiVisionFmTorchBaseline, optimizer: Any, metrics: dict[str, float], config: dict[str, Any]) -> Path:
    torch = require_torch_backend()
    path = Path(output_dir) / "checkpoint_last.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        path,
    )
    return path


def train_one_epoch(model: FwiVisionFmTorchBaseline, dataloader: Any, optimizer: Any, *, device: str = "cpu") -> dict[str, float]:
    torch = require_torch_backend()
    criterion = torch.nn.MSELoss()
    model.train()
    losses = []
    predictions = []
    targets = []
    for batch in dataloader:
        records = batch["records"].to(device)
        velocity = batch["velocity"].to(device)
        source_positions = batch["source_positions"].to(device)
        optimizer.zero_grad()
        prediction = model(records, source_positions)
        loss = criterion(prediction, velocity)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        predictions.append(prediction.detach().cpu().numpy())
        targets.append(velocity.detach().cpu().numpy())
    metrics = velocity_metrics(np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0))
    return {"loss": float(np.mean(losses)), **metrics}


def evaluate(model: FwiVisionFmTorchBaseline, dataloader: Any, *, device: str = "cpu") -> dict[str, float]:
    torch = require_torch_backend()
    criterion = torch.nn.MSELoss()
    model.eval()
    losses = []
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in dataloader:
            records = batch["records"].to(device)
            velocity = batch["velocity"].to(device)
            source_positions = batch["source_positions"].to(device)
            prediction = model(records, source_positions)
            loss = criterion(prediction, velocity)
            losses.append(float(loss.detach().cpu()))
            predictions.append(prediction.detach().cpu().numpy())
            targets.append(velocity.detach().cpu().numpy())
    if not losses:
        return {
            "loss": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "relative_mae": 0.0,
            "relative_rmse": 0.0,
            "psnr": 0.0,
            "ssim": 0.0,
            "gradient_error": 0.0,
        }
    metrics = velocity_metrics(np.concatenate(predictions, axis=0), np.concatenate(targets, axis=0))
    return {"loss": float(np.mean(losses)), **metrics}


def _save_prediction_samples(model: FwiVisionFmTorchBaseline, dataloader: Any, output_dir: Path, *, device: str, limit: int = 3) -> None:
    torch = require_torch_backend()
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            prediction = model(batch["records"].to(device), batch["source_positions"].to(device)).cpu().numpy()
            target = batch["velocity"].cpu().numpy()
            for index in range(prediction.shape[0]):
                if saved >= limit:
                    return
                np.savez(
                    pred_dir / f"sample_{saved:03d}.npz",
                    velocity_true=target[index].astype(np.float32),
                    velocity_pred=prediction[index].astype(np.float32),
                    velocity_error=(prediction[index] - target[index]).astype(np.float32),
                )
                saved += 1


def _save_prediction_arrays(model: FwiVisionFmTorchBaseline, dataloader: Any, output_dir: Path, *, device: str) -> None:
    torch = require_torch_backend()
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            records = batch["records"].to(device)
            source_positions = batch["source_positions"].to(device)
            prediction = model(records, source_positions).cpu().numpy()
            target = batch["velocity"].cpu().numpy()
            input_seismic = batch["records"].cpu().numpy()
            np.save(output_dir / "prediction.npy", prediction[0].astype(np.float32))
            np.save(output_dir / "target.npy", target[0].astype(np.float32))
            np.save(output_dir / "input_seismic.npy", input_seismic[0].astype(np.float32))
            return


def run_torch_npz_experiment(
    *,
    data_dir: str | Path | None,
    output_dir: str | Path,
    depth: int,
    width: int,
    channels: tuple[str, ...] = ("raw",),
    bridge_mode: str = "simple",
    aggregation: str = "mean",
    decoder_mode: str = "bounded",
    batch_size: int = 4,
    epochs: int = 5,
    learning_rate: float = 1.0e-3,
    vmin: float = 1500.0,
    vmax: float = 4500.0,
    device: str = "cpu",
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 0,
    split_paths: dict[str, list[str | Path]] | None = None,
    save_prediction_samples: bool = True,
) -> dict[str, Any]:
    torch = require_torch_backend()
    set_torch_seed(seed)
    if split_paths is None:
        if data_dir is None:
            raise ValueError("data_dir is required when split_paths is not provided")
        paths = discover_npz_samples(data_dir)
        if not paths:
            raise ValueError(f"no npz samples found in {data_dir}")
        split = split_sample_paths(paths, train_fraction=train_fraction, val_fraction=val_fraction, seed=seed)
    else:
        split = {name: [Path(path) for path in split_paths.get(name, [])] for name in ("train", "val", "test")}
        paths = [path for items in split.values() for path in items]
    if not split["train"]:
        raise ValueError("train split is empty")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "data_dir": "" if data_dir is None else str(Path(data_dir)),
        "depth": int(depth),
        "width": int(width),
        "channels": list(channels),
        "bridge_mode": bridge_mode,
        "aggregation": aggregation,
        "decoder_mode": decoder_mode,
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "vmin": float(vmin),
        "vmax": float(vmax),
        "device": device,
        "train_fraction": float(train_fraction),
        "val_fraction": float(val_fraction),
        "seed": int(seed),
        "used_split_manifest": split_paths is not None,
    }
    (output / "resolved_torch_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    train_loader = build_torch_dataloader(split["train"], batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = build_torch_dataloader(split["val"] or split["train"], batch_size=batch_size, shuffle=False, seed=seed)
    test_loader = build_torch_dataloader(split["test"] or split["train"], batch_size=batch_size, shuffle=False, seed=seed)
    model = FwiVisionFmTorchBaseline(
        channels=channels,
        depth=depth,
        width=width,
        bridge_mode=bridge_mode,
        aggregation=aggregation,
        decoder_mode=decoder_mode,
        vmin=vmin,
        vmax=vmax,
    ).to(device)
    parameter_counts = count_model_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    latest_metrics = {}
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device=device)
        val_metrics = evaluate(model, val_loader, device=device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_relative_mae": train_metrics["relative_mae"],
            "train_relative_rmse": train_metrics["relative_rmse"],
            "train_psnr": train_metrics["psnr"],
            "train_ssim": train_metrics["ssim"],
            "train_gradient_error": train_metrics["gradient_error"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_relative_mae": val_metrics["relative_mae"],
            "val_relative_rmse": val_metrics["relative_rmse"],
            "val_psnr": val_metrics["psnr"],
            "val_ssim": val_metrics["ssim"],
            "val_gradient_error": val_metrics["gradient_error"],
        }
        history.append(row)
        save_checkpoint(output, epoch, model, optimizer, row)
        latest_metrics = row
    test_metrics = evaluate(model, test_loader, device=device)
    if save_prediction_samples:
        _save_prediction_samples(model, test_loader, output, device=device)
    save_last_checkpoint(output, model, optimizer, {**latest_metrics, **{"test_loss": test_metrics["loss"], "test_mae": test_metrics["mae"], "test_rmse": test_metrics["rmse"]}}, config)
    (output / "torch_training_history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "torch_training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "model_type": "torch_cnn_baseline",
        "backbone_name": "simple_cnn",
        "pretrained": False,
        "freeze_backbone": False,
        "bridge": bridge_mode,
        "aggregation": aggregation,
        "decoder": decoder_mode,
        "depth": int(depth),
        "width": int(width),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "trainable_parameters": parameter_counts["trainable_parameters"],
        "total_parameters": parameter_counts["total_parameters"],
        "trainable_ratio": parameter_counts["trainable_ratio"],
        "device": device,
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "final_train_mae": history[-1]["train_mae"],
        "final_val_mae": history[-1]["val_mae"],
        "final_train_rmse": history[-1]["train_rmse"],
        "final_val_rmse": history[-1]["val_rmse"],
        "final_train_relative_mae": history[-1]["train_relative_mae"],
        "final_val_relative_mae": history[-1]["val_relative_mae"],
        "final_train_relative_rmse": history[-1]["train_relative_rmse"],
        "final_val_relative_rmse": history[-1]["val_relative_rmse"],
        "final_train_psnr": history[-1]["train_psnr"],
        "final_val_psnr": history[-1]["val_psnr"],
        "final_train_ssim": history[-1]["train_ssim"],
        "final_val_ssim": history[-1]["val_ssim"],
        "final_train_gradient_error": history[-1]["train_gradient_error"],
        "final_val_gradient_error": history[-1]["val_gradient_error"],
        "epochs": int(epochs),
        "sample_count": len(paths),
        "split_counts": {name: len(items) for name, items in split.items()},
        "final_train": history[-1],
        "test_metrics": test_metrics,
        "config": config,
    }
    (output / "torch_experiment_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def run_torch_smoke_experiment(
    output_dir: str | Path,
    *,
    samples: int = 4,
    shots: int = 3,
    receivers: int = 8,
    time_samples: int = 12,
    depth: int = 6,
    width: int = 7,
    channels: tuple[str, ...] = ("raw", "offset"),
    aggregation: str = "source_attention",
    batch_size: int = 2,
    epochs: int = 1,
    learning_rate: float = 1.0e-3,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    data_dir = output_path / "torch_smoke_inputs"
    cfg = DataConfig(
        num_shots=shots,
        num_receivers=receivers,
        num_time_samples=time_samples,
        velocity_depth=depth,
        velocity_width=width,
    )
    records = []
    velocity = []
    source_positions = []
    for index in range(samples):
        sample = make_synthetic_sample(cfg, seed=seed + index)
        records.append(sample.records)
        velocity.append(sample.velocity)
        source_positions.append(sample.source_positions)
    convert_array_dataset_to_npz(
        np.stack(records, axis=0),
        np.stack(velocity, axis=0),
        data_dir,
        dataset_name="torch_smoke",
        source_positions=np.stack(source_positions, axis=0),
    )
    return run_torch_npz_experiment(
        data_dir=data_dir,
        output_dir=output_path,
        depth=depth,
        width=width,
        channels=channels,
        bridge_mode="simple",
        aggregation=aggregation,
        decoder_mode="bounded",
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
    )


def run_torch_cpu_experiment(
    output_dir: str | Path,
    *,
    samples: int = 6,
    shots: int = 3,
    receivers: int = 8,
    time_samples: int = 12,
    depth: int = 6,
    width: int = 7,
    channels: tuple[str, ...] = ("raw", "offset"),
    aggregation: str = "source_attention",
    batch_size: int = 2,
    epochs: int = 3,
    learning_rate: float = 1.0e-3,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    data_dir = output_path / "torch_cpu_experiment_inputs"
    cfg = DataConfig(
        num_shots=shots,
        num_receivers=receivers,
        num_time_samples=time_samples,
        velocity_depth=depth,
        velocity_width=width,
    )
    records = []
    velocity = []
    source_positions = []
    for index in range(samples):
        sample = make_synthetic_sample(cfg, seed=seed + index)
        records.append(sample.records)
        velocity.append(sample.velocity)
        source_positions.append(sample.source_positions)
    convert_array_dataset_to_npz(
        np.stack(records, axis=0),
        np.stack(velocity, axis=0),
        data_dir,
        dataset_name="torch_cpu_experiment",
        source_positions=np.stack(source_positions, axis=0),
    )
    summary = run_torch_npz_experiment(
        data_dir=data_dir,
        output_dir=output_path,
        depth=depth,
        width=width,
        channels=channels,
        bridge_mode="simple",
        aggregation=aggregation,
        decoder_mode="bounded",
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        seed=seed,
    )
    metrics = {
        "final_train_loss": summary["final_train_loss"],
        "final_val_loss": summary["final_val_loss"],
        "final_train_mae": summary["final_train_mae"],
        "final_val_mae": summary["final_val_mae"],
        "final_train_rmse": summary["final_train_rmse"],
        "final_val_rmse": summary["final_val_rmse"],
        "final_train_relative_mae": summary.get("final_train_relative_mae"),
        "final_val_relative_mae": summary.get("final_val_relative_mae"),
        "final_train_relative_rmse": summary.get("final_train_relative_rmse"),
        "final_val_relative_rmse": summary.get("final_val_relative_rmse"),
        "final_train_psnr": summary.get("final_train_psnr"),
        "final_val_psnr": summary.get("final_val_psnr"),
        "final_train_ssim": summary.get("final_train_ssim"),
        "final_val_ssim": summary.get("final_val_ssim"),
        "final_train_gradient_error": summary.get("final_train_gradient_error"),
        "final_val_gradient_error": summary.get("final_val_gradient_error"),
        "test_loss": summary["test_metrics"]["loss"],
        "test_mae": summary["test_metrics"]["mae"],
        "test_rmse": summary["test_metrics"]["rmse"],
        "test_relative_mae": summary["test_metrics"].get("relative_mae"),
        "test_relative_rmse": summary["test_metrics"].get("relative_rmse"),
        "test_psnr": summary["test_metrics"].get("psnr"),
        "test_ssim": summary["test_metrics"].get("ssim"),
        "test_gradient_error": summary["test_metrics"].get("gradient_error"),
    }
    experiment_summary = {
        **summary,
        "experiment_type": "torch_cpu_experiment",
        "cpu_only": True,
        "config": {
            **summary.get("config", {}),
            "samples": int(samples),
            "shots": int(shots),
            "receivers": int(receivers),
            "time_samples": int(time_samples),
            "seed": int(seed),
            "sample_count": int(samples),
        },
    }
    history_csv = output_path / "torch_training_history.csv"
    if history_csv.exists():
        (output_path / "training_history.csv").write_text(history_csv.read_text(encoding="utf-8"), encoding="utf-8")
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "experiment_summary.json").write_text(
        json.dumps(experiment_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    test_loader = build_torch_dataloader(discover_npz_samples(data_dir), batch_size=batch_size, shuffle=False, seed=seed)
    checkpoint = Path(output_path) / "checkpoint_last.pt"
    torch = require_torch_backend()
    model = FwiVisionFmTorchBaseline(
        channels=channels,
        depth=depth,
        width=width,
        aggregation=aggregation,
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    _save_prediction_arrays(model, test_loader, output_path, device=device)
    return experiment_summary


def run_torch_ablation_experiment(
    output_dir: str | Path,
    *,
    samples: int = 6,
    shots: int = 3,
    receivers: int = 8,
    time_samples: int = 12,
    depth: int = 6,
    width: int = 7,
    channels: tuple[str, ...] = ("raw", "offset"),
    batch_size: int = 2,
    epochs: int = 1,
    learning_rate: float = 1.0e-3,
    seed: int = 0,
    device: str = "cpu",
) -> dict[str, Any]:
    output_path = Path(output_dir)
    data_dir = output_path / "torch_ablation_inputs"
    cfg = DataConfig(
        num_shots=shots,
        num_receivers=receivers,
        num_time_samples=time_samples,
        velocity_depth=depth,
        velocity_width=width,
    )
    records = []
    velocity = []
    source_positions = []
    for index in range(samples):
        sample = make_synthetic_sample(cfg, seed=seed + index)
        records.append(sample.records)
        velocity.append(sample.velocity)
        source_positions.append(sample.source_positions)
    convert_array_dataset_to_npz(
        np.stack(records, axis=0),
        np.stack(velocity, axis=0),
        data_dir,
        dataset_name="torch_ablation",
        source_positions=np.stack(source_positions, axis=0),
    )

    runs_root = output_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    bridge_modes = ("simple", "normalized", "channel_stack")
    aggregations = ("mean", "max")
    decoder_modes = ("bounded", "unbounded")
    best_row: dict[str, Any] | None = None

    for bridge_mode in bridge_modes:
        for aggregation in aggregations:
            for decoder_mode in decoder_modes:
                experiment_id = f"{bridge_mode}_{aggregation}_{decoder_mode}"
                run_dir = runs_root / experiment_id
                summary = run_torch_npz_experiment(
                    data_dir=data_dir,
                    output_dir=run_dir,
                    depth=depth,
                    width=width,
                    channels=channels,
                    bridge_mode=bridge_mode,
                    aggregation=aggregation,
                    decoder_mode=decoder_mode,
                    batch_size=batch_size,
                    epochs=epochs,
                    learning_rate=learning_rate,
                    device=device,
                    seed=seed,
                    save_prediction_samples=False,
                )
                history_csv = run_dir / "torch_training_history.csv"
                if history_csv.exists():
                    (run_dir / "training_history.csv").write_text(history_csv.read_text(encoding="utf-8"), encoding="utf-8")
                metrics = {
                    "final_val_loss": summary["final_val_loss"],
                    "final_val_mae": summary["final_val_mae"],
                    "final_val_rmse": summary["final_val_rmse"],
                    "final_val_relative_mae": summary.get("final_val_relative_mae"),
                    "final_val_relative_rmse": summary.get("final_val_relative_rmse"),
                    "final_val_psnr": summary.get("final_val_psnr"),
                    "final_val_ssim": summary.get("final_val_ssim"),
                    "final_val_gradient_error": summary.get("final_val_gradient_error"),
                    "test_mae": summary["test_metrics"]["mae"],
                    "test_rmse": summary["test_metrics"]["rmse"],
                    "test_relative_mae": summary["test_metrics"].get("relative_mae"),
                    "test_relative_rmse": summary["test_metrics"].get("relative_rmse"),
                    "test_psnr": summary["test_metrics"].get("psnr"),
                    "test_ssim": summary["test_metrics"].get("ssim"),
                    "test_gradient_error": summary["test_metrics"].get("gradient_error"),
                }
                (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
                (run_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
                row = {
                    "experiment_id": experiment_id,
                    "bridge": bridge_mode,
                    "aggregation": aggregation,
                    "decoder": decoder_mode,
                    "epochs": int(epochs),
                    "final_val_loss": summary["final_val_loss"],
                    "final_val_mae": summary["final_val_mae"],
                    "final_val_rmse": summary["final_val_rmse"],
                    "final_val_relative_mae": summary.get("final_val_relative_mae"),
                    "final_val_relative_rmse": summary.get("final_val_relative_rmse"),
                    "final_val_psnr": summary.get("final_val_psnr"),
                    "final_val_ssim": summary.get("final_val_ssim"),
                    "final_val_gradient_error": summary.get("final_val_gradient_error"),
                    "test_mae": summary["test_metrics"]["mae"],
                    "test_rmse": summary["test_metrics"]["rmse"],
                    "test_relative_mae": summary["test_metrics"].get("relative_mae"),
                    "test_relative_rmse": summary["test_metrics"].get("relative_rmse"),
                    "test_psnr": summary["test_metrics"].get("psnr"),
                    "test_ssim": summary["test_metrics"].get("ssim"),
                    "test_gradient_error": summary["test_metrics"].get("gradient_error"),
                    "output_dir": str(run_dir),
                }
                rows.append(row)
                if best_row is None or float(row["test_mae"]) < float(best_row["test_mae"]):
                    best_row = row

    if best_row is None:
        raise RuntimeError("ablation produced no experiment rows")

    with (output_path / "ablation_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment_id",
                "bridge",
                "aggregation",
                "decoder",
                "epochs",
                "final_val_loss",
                "final_val_mae",
                "final_val_rmse",
                "final_val_relative_mae",
                "final_val_relative_rmse",
                "final_val_psnr",
                "final_val_ssim",
                "final_val_gradient_error",
                "test_mae",
                "test_rmse",
                "test_relative_mae",
                "test_relative_rmse",
                "test_psnr",
                "test_ssim",
                "test_gradient_error",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    best_run_dir = Path(best_row["output_dir"])
    best_summary = json.loads((best_run_dir / "experiment_summary.json").read_text(encoding="utf-8"))
    test_loader = build_torch_dataloader(discover_npz_samples(data_dir), batch_size=batch_size, shuffle=False, seed=seed)
    checkpoint = best_run_dir / "checkpoint_last.pt"
    torch = require_torch_backend()
    model = FwiVisionFmTorchBaseline(
        channels=channels,
        depth=depth,
        width=width,
        bridge_mode=str(best_row["bridge"]),
        aggregation=str(best_row["aggregation"]),
        decoder_mode=str(best_row["decoder"]),
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    _save_prediction_arrays(model, test_loader, best_run_dir, device=device)

    ablation_summary = {
        "experiment_count": len(rows),
        "device": device,
        "epochs": int(epochs),
        "best_metric": "test_mae",
        "best_experiment_id": best_row["experiment_id"],
        "best_output_dir": str(best_run_dir),
        "results": rows,
    }
    (output_path / "ablation_summary.json").write_text(json.dumps(ablation_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "best_config.json").write_text(
        json.dumps({**best_row, "config": best_summary.get("config", {})}, indent=2, ensure_ascii=False), encoding="utf-8")
    return ablation_summary


def _resolve_openfwi_split_file(split_dir: str | Path, candidates: tuple[str, ...]) -> Path:
    root = Path(split_dir)
    for name in candidates:
        candidate = root / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing split file under {root}; tried: {', '.join(candidates)}")


def _stack_openfwi_dataset(
    *,
    root: str | Path,
    split_file: str | Path,
    stats_file: str | Path,
    max_samples: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = OpenFWINpyDataset(
        root=str(root),
        split_file=str(split_file),
        max_samples=max_samples,
        input_norm="zscore",
        target_norm="none",
        stats_file=str(stats_file),
        fit_stats=False,
    )
    records_list: list[np.ndarray] = []
    velocity_list: list[np.ndarray] = []
    source_positions_list: list[np.ndarray] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        seismic = sample["seismic"].numpy().astype(np.float32, copy=False)
        velocity = sample["velocity"].numpy().astype(np.float32, copy=False)
        if seismic.ndim != 3:
            raise ValueError(f"expected OpenFWI seismic sample to be 3D, got {seismic.shape}")
        if velocity.ndim != 3 or velocity.shape[0] != 1:
            raise ValueError(f"expected OpenFWI velocity sample to be [1,H,W], got {velocity.shape}")
        shots = int(seismic.shape[0])
        records_list.append(np.transpose(seismic, (0, 2, 1)).astype(np.float32, copy=False))
        velocity_list.append(velocity[0].astype(np.float32, copy=False))
        source_positions_list.append(np.linspace(0.12, 0.88, shots, dtype=np.float32))
    if not records_list:
        raise ValueError(f"no samples loaded from split {split_file}")
    return (
        np.stack(records_list, axis=0),
        np.stack(velocity_list, axis=0),
        np.stack(source_positions_list, axis=0),
    )


def _write_openfwi_npz_split(
    *,
    data_root: str | Path,
    split_file: str | Path,
    stats_file: str | Path,
    output_dir: Path,
    split_name: str,
    max_samples: int | None,
) -> list[Path]:
    records, velocity, source_positions = _stack_openfwi_dataset(
        root=data_root,
        split_file=split_file,
        stats_file=stats_file,
        max_samples=max_samples,
    )
    manifest = convert_array_dataset_to_npz(
        records,
        velocity,
        output_dir,
        dataset_name=f"openfwi_small_{split_name}",
        source_positions=source_positions,
    )
    return [output_dir / sample["path"] for sample in manifest["samples"]]


def _read_openfwi_velocity_bounds(stats_file: str | Path) -> tuple[float, float]:
    payload = json.loads(Path(stats_file).read_text(encoding="utf-8"))
    if "velocity" in payload:
        velocity = payload["velocity"]
        if "min" in velocity and "max" in velocity:
            return float(velocity["min"]), float(velocity["max"])
    if "target_min" in payload and "target_max" in payload:
        return float(payload["target_min"]), float(payload["target_max"])
    return 1500.0, 4500.0


def run_openfwi_small_experiment(
    output_dir: str | Path,
    *,
    data_root: str | Path,
    split_dir: str | Path,
    stats_file: str | Path,
    epochs: int = 2,
    batch_size: int = 1,
    seed: int = 0,
    device: str = "cpu",
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
    learning_rate: float = 1.0e-3,
) -> dict[str, Any]:
    data_root_path = Path(data_root)
    split_dir_path = Path(split_dir)
    stats_file_path = Path(stats_file)
    if not data_root_path.exists():
        raise FileNotFoundError(f"OpenFWI data-root does not exist: {data_root_path}")
    if not split_dir_path.exists():
        raise FileNotFoundError(f"OpenFWI split-dir does not exist: {split_dir_path}")
    if not stats_file_path.exists():
        raise FileNotFoundError(f"OpenFWI stats-file does not exist: {stats_file_path}")

    output_path = Path(output_dir)
    npz_root = output_path / "openfwi_small_inputs"
    train_split = _resolve_openfwi_split_file(split_dir_path, ("train.csv",))
    val_split = _resolve_openfwi_split_file(split_dir_path, ("val.csv", "smoke_val.csv"))
    test_split = _resolve_openfwi_split_file(split_dir_path, ("test_in_family.csv", "test.csv", "smoke_val.csv"))

    train_paths = _write_openfwi_npz_split(
        data_root=data_root_path,
        split_file=train_split,
        stats_file=stats_file_path,
        output_dir=npz_root / "train",
        split_name="train",
        max_samples=max_train_samples,
    )
    val_paths = _write_openfwi_npz_split(
        data_root=data_root_path,
        split_file=val_split,
        stats_file=stats_file_path,
        output_dir=npz_root / "val",
        split_name="val",
        max_samples=max_val_samples,
    )
    test_paths = _write_openfwi_npz_split(
        data_root=data_root_path,
        split_file=test_split,
        stats_file=stats_file_path,
        output_dir=npz_root / "test",
        split_name="test",
        max_samples=max_test_samples,
    )

    first_sample = np.load(train_paths[0])
    depth, width = map(int, first_sample["velocity"].shape)
    shots, receivers, time_samples = map(int, first_sample["records"].shape)
    first_sample.close()
    vmin, vmax = _read_openfwi_velocity_bounds(stats_file_path)
    summary = run_torch_npz_experiment(
        data_dir=npz_root,
        output_dir=output_path,
        depth=depth,
        width=width,
        channels=("raw", "offset"),
        bridge_mode="channel_stack",
        aggregation="max",
        decoder_mode="bounded",
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        vmin=vmin,
        vmax=vmax,
        device=device,
        seed=seed,
        split_paths={"train": train_paths, "val": val_paths, "test": test_paths},
        save_prediction_samples=False,
    )
    metrics = {
        "final_train_loss": summary["final_train_loss"],
        "final_val_loss": summary["final_val_loss"],
        "final_train_mae": summary["final_train_mae"],
        "final_val_mae": summary["final_val_mae"],
        "final_train_rmse": summary["final_train_rmse"],
        "final_val_rmse": summary["final_val_rmse"],
        "final_train_relative_mae": summary.get("final_train_relative_mae"),
        "final_val_relative_mae": summary.get("final_val_relative_mae"),
        "final_train_relative_rmse": summary.get("final_train_relative_rmse"),
        "final_val_relative_rmse": summary.get("final_val_relative_rmse"),
        "final_train_psnr": summary.get("final_train_psnr"),
        "final_val_psnr": summary.get("final_val_psnr"),
        "final_train_ssim": summary.get("final_train_ssim"),
        "final_val_ssim": summary.get("final_val_ssim"),
        "final_train_gradient_error": summary.get("final_train_gradient_error"),
        "final_val_gradient_error": summary.get("final_val_gradient_error"),
        "test_loss": summary["test_metrics"]["loss"],
        "test_mae": summary["test_metrics"]["mae"],
        "test_rmse": summary["test_metrics"]["rmse"],
        "test_relative_mae": summary["test_metrics"].get("relative_mae"),
        "test_relative_rmse": summary["test_metrics"].get("relative_rmse"),
        "test_psnr": summary["test_metrics"].get("psnr"),
        "test_ssim": summary["test_metrics"].get("ssim"),
        "test_gradient_error": summary["test_metrics"].get("gradient_error"),
    }
    history_csv = output_path / "torch_training_history.csv"
    if history_csv.exists():
        (output_path / "training_history.csv").write_text(history_csv.read_text(encoding="utf-8"), encoding="utf-8")
    experiment_summary = {
        **summary,
        "experiment_type": "openfwi_small_experiment",
        "data_type": "openfwi_small",
        "bridge": "channel_stack",
        "aggregation": "max",
        "decoder": "bounded",
        "config": {
            **summary.get("config", {}),
            "data_root": str(data_root_path),
            "split_dir": str(split_dir_path),
            "stats_file": str(stats_file_path),
            "sample_count": int(len(train_paths) + len(val_paths) + len(test_paths)),
            "shots": shots,
            "receivers": receivers,
            "time_samples": time_samples,
            "max_train_samples": max_train_samples,
            "max_val_samples": max_val_samples,
            "max_test_samples": max_test_samples,
        },
    }
    (output_path / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "experiment_summary.json").write_text(
        json.dumps(experiment_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    checkpoint = output_path / "checkpoint_last.pt"
    torch = require_torch_backend()
    model = FwiVisionFmTorchBaseline(
        channels=("raw", "offset"),
        depth=depth,
        width=width,
        bridge_mode="channel_stack",
        aggregation="max",
        decoder_mode="bounded",
        vmin=vmin,
        vmax=vmax,
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    test_loader = build_torch_dataloader(test_paths, batch_size=batch_size, shuffle=False, seed=seed)
    _save_prediction_arrays(model, test_loader, output_path, device=device)
    return experiment_summary


def parse_scale_sizes(spec: str) -> list[tuple[int, int, int]]:
    values: list[tuple[int, int, int]] = []
    for raw_part in str(spec).split(","):
        part = raw_part.strip()
        if not part:
            continue
        tokens = part.split(":")
        if len(tokens) != 3:
            raise ValueError(f"invalid size triple: {part}")
        train_count, val_count, test_count = (int(token) for token in tokens)
        if train_count <= 0 or val_count <= 0 or test_count <= 0:
            raise ValueError(f"size triple must be positive: {part}")
        values.append((train_count, val_count, test_count))
    if not values:
        raise ValueError("sizes must contain at least one train:val:test triple")
    return values


def run_openfwi_scale_study(
    output_dir: str | Path,
    *,
    data_root: str | Path,
    split_dir: str | Path,
    stats_file: str | Path,
    sizes: str = "8:2:2,32:8:8,64:8:8",
    epochs: int = 3,
    batch_size: int = 1,
    seed: int = 0,
    device: str = "cpu",
    learning_rate: float = 1.0e-3,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    runs_root = output_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    size_triples = parse_scale_sizes(sizes)
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    max_train = max(train for train, _, _ in size_triples)

    for train_count, val_count, test_count in size_triples:
        experiment_id = f"train{train_count}_val{val_count}_test{test_count}"
        run_dir = runs_root / experiment_id
        summary = run_openfwi_small_experiment(
            run_dir,
            data_root=data_root,
            split_dir=split_dir,
            stats_file=stats_file,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            device=device,
            max_train_samples=train_count,
            max_val_samples=val_count,
            max_test_samples=test_count,
            learning_rate=learning_rate,
        )
        row = {
            "experiment_id": experiment_id,
            "train_samples": int(train_count),
            "val_samples": int(val_count),
            "test_samples": int(test_count),
            "epochs": int(epochs),
            "bridge": "channel_stack",
            "aggregation": "max",
            "decoder": "bounded",
            "final_val_loss": float(summary["final_val_loss"]),
            "final_val_mae": float(summary["final_val_mae"]),
            "final_val_rmse": float(summary["final_val_rmse"]),
            "final_val_relative_mae": summary.get("final_val_relative_mae"),
            "final_val_relative_rmse": summary.get("final_val_relative_rmse"),
            "final_val_psnr": summary.get("final_val_psnr"),
            "final_val_ssim": summary.get("final_val_ssim"),
            "final_val_gradient_error": summary.get("final_val_gradient_error"),
            "test_mae": float(summary["test_metrics"]["mae"]),
            "test_rmse": float(summary["test_metrics"]["rmse"]),
            "test_relative_mae": summary["test_metrics"].get("relative_mae"),
            "test_relative_rmse": summary["test_metrics"].get("relative_rmse"),
            "test_psnr": summary["test_metrics"].get("psnr"),
            "test_ssim": summary["test_metrics"].get("ssim"),
            "test_gradient_error": summary["test_metrics"].get("gradient_error"),
            "output_dir": str(run_dir),
        }
        rows.append(row)
        if best_row is None or float(row["test_mae"]) < float(best_row["test_mae"]):
            best_row = row

    if best_row is None:
        raise RuntimeError("scale study produced no experiment rows")

    keep_array_dirs = {
        str(Path(best_row["output_dir"])),
        str(runs_root / f"train{max_train}_val{next(val for train, val, test in size_triples if train == max_train)}_test{next(test for train, val, test in size_triples if train == max_train)}"),
    }
    for row in rows:
        run_dir = Path(str(row["output_dir"]))
        if str(run_dir) in keep_array_dirs:
            continue
        for file_name in ("prediction.npy", "target.npy", "input_seismic.npy", "loss_curve.png", "prediction_vs_target.png", "report.md"):
            path = run_dir / file_name
            if path.exists():
                path.unlink()

    with (output_path / "scale_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment_id",
                "train_samples",
                "val_samples",
                "test_samples",
                "epochs",
                "bridge",
                "aggregation",
                "decoder",
                "final_val_loss",
                "final_val_mae",
                "final_val_rmse",
                "final_val_relative_mae",
                "final_val_relative_rmse",
                "final_val_psnr",
                "final_val_ssim",
                "final_val_gradient_error",
                "test_mae",
                "test_rmse",
                "test_relative_mae",
                "test_relative_rmse",
                "test_psnr",
                "test_ssim",
                "test_gradient_error",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_payload = {
        "experiment_count": len(rows),
        "device": device,
        "epochs": int(epochs),
        "sizes": [{"train": train, "val": val, "test": test} for train, val, test in size_triples],
        "best_metric": "test_mae",
        "best_experiment_id": best_row["experiment_id"],
        "results": rows,
    }
    (output_path / "scale_summary.json").write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_path / "best_scale_config.json").write_text(json.dumps(best_row, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_payload
