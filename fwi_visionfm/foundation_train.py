from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.config import DataConfig
from fwi_visionfm.data_conversion import convert_array_dataset_to_npz
from fwi_visionfm.data.openfwi_npy_dataset import OpenFWINpyDataset
from fwi_visionfm.datasets import discover_npz_samples, make_synthetic_sample, split_sample_paths
from fwi_visionfm.evaluation.metrics import compute_velocity_metrics
from fwi_visionfm.models.parameter_utils import count_parameters
from fwi_visionfm.peft import AdapterConfig, LoRAConfig, count_trainable_parameters
from fwi_visionfm.torch_backend import require_torch_backend
from fwi_visionfm.torch_backend.data import build_torch_dataloader
from fwi_visionfm.torch_backend.model import FrozenFoundationFWI
from fwi_visionfm.torch_backend.train import set_torch_seed, torch_metrics


def _extract_batch(batch: Any, *, device: str) -> tuple[Any, Any, Any, Any]:
    records = batch["records"] if "records" in batch else batch["seismic"]
    velocity = batch["velocity"]
    source_positions = batch.get("source_positions")
    meta = batch.get("meta")
    if velocity.ndim == 4 and velocity.shape[1] == 1:
        velocity = velocity[:, 0]
    records = records.to(device)
    velocity = velocity.to(device)
    if source_positions is not None:
        source_positions = source_positions.to(device)
    return records, velocity, source_positions, meta


def _save_prediction_samples(model: FrozenFoundationFWI, dataloader: Any, output_dir: Path, *, device: str, limit: int = 3) -> None:
    torch = require_torch_backend()
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            records, velocity, source_positions, _ = _extract_batch(batch, device=device)
            prediction = model(records, source_positions).cpu().numpy()
            target = velocity.cpu().numpy()
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


def _save_predictions_preview(model: FrozenFoundationFWI, dataloader: Any, output_path: Path, *, device: str, limit: int = 8) -> None:
    torch = require_torch_backend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    collected_records = []
    collected_true = []
    collected_pred = []
    collected_error = []
    collected_meta = []
    saved = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            records, velocity, source_positions, meta = _extract_batch(batch, device=device)
            prediction = model(records, source_positions).cpu().numpy()
            records_np = records.cpu().numpy()
            velocity_np = velocity.cpu().numpy()
            meta_items = _normalize_meta_batch(meta, prediction.shape[0])
            for index in range(prediction.shape[0]):
                if saved >= limit:
                    np.savez(
                        output_path,
                        seismic_preview=np.asarray(collected_records, dtype=np.float32),
                        velocity_true=np.asarray(collected_true, dtype=np.float32),
                        velocity_pred=np.asarray(collected_pred, dtype=np.float32),
                        error_map=np.asarray(collected_error, dtype=np.float32),
                        meta=np.asarray(collected_meta, dtype=object),
                    )
                    return
                collected_records.append(records_np[index].astype(np.float32))
                collected_true.append(velocity_np[index].astype(np.float32))
                collected_pred.append(prediction[index].astype(np.float32))
                collected_error.append((prediction[index] - velocity_np[index]).astype(np.float32))
                collected_meta.append(json.dumps(meta_items[index], ensure_ascii=False))
                saved += 1
    np.savez(
        output_path,
        seismic_preview=np.asarray(collected_records, dtype=np.float32),
        velocity_true=np.asarray(collected_true, dtype=np.float32),
        velocity_pred=np.asarray(collected_pred, dtype=np.float32),
        error_map=np.asarray(collected_error, dtype=np.float32),
        meta=np.asarray(collected_meta, dtype=object),
    )


def _normalize_meta_batch(meta: Any, batch_size: int) -> list[dict[str, Any]]:
    def _to_json_safe(value: Any) -> Any:
        if hasattr(value, "item") and callable(value.item):
            try:
                return value.item()
            except (ValueError, RuntimeError):
                pass
        if hasattr(value, "tolist") and callable(value.tolist):
            try:
                return value.tolist()
            except (ValueError, RuntimeError):
                pass
        return value

    if meta is None:
        return [{} for _ in range(batch_size)]
    if isinstance(meta, list):
        return [dict(item) if isinstance(item, dict) else {"value": _to_json_safe(item)} for item in meta]
    if isinstance(meta, dict):
        normalized = []
        for index in range(batch_size):
            item = {}
            for key, value in meta.items():
                if isinstance(value, list):
                    item[key] = _to_json_safe(value[index])
                else:
                    item[key] = _to_json_safe(value)
            normalized.append(item)
        return normalized
    return [{"value": str(_to_json_safe(meta))} for _ in range(batch_size)]


def _resolve_backbone_identity(
    foundation_backbone: str,
    backbone_type: str | None,
    model_name: str | None,
) -> tuple[str, str]:
    if backbone_type is not None:
        return str(backbone_type), str(model_name or ("dummy_dinov2" if backbone_type == "dummy" else foundation_backbone))
    if foundation_backbone == "dummy_dinov2":
        return "dummy", "dummy_dinov2"
    if str(model_name or foundation_backbone).startswith("facebook/"):
        return "hf_dinov2", str(model_name or foundation_backbone)
    return "timm", str(model_name or foundation_backbone)


def _trainable_state_dict(model: FrozenFoundationFWI) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_foundation_checkpoint(output_dir: str | Path, model: FrozenFoundationFWI, optimizer: Any, config: dict[str, Any], metrics: dict[str, float]) -> Path:
    torch = require_torch_backend()
    path = Path(output_dir) / "checkpoint_last.pt"
    torch.save(
        {
            "trainable_state_dict": _trainable_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "metrics": metrics,
        },
        path,
    )
    return path


def train_one_epoch(model: FrozenFoundationFWI, dataloader: Any, optimizer: Any, *, device: str = "cpu") -> dict[str, float]:
    torch = require_torch_backend()
    criterion = torch.nn.MSELoss()
    model.train()
    losses = []
    maes = []
    rmses = []
    for batch in dataloader:
        records, velocity, source_positions, _ = _extract_batch(batch, device=device)
        optimizer.zero_grad()
        prediction = model(records, source_positions)
        loss = criterion(prediction, velocity)
        loss.backward()
        optimizer.step()
        metrics = torch_metrics(prediction, velocity)
        losses.append(float(loss.detach().cpu()))
        maes.append(metrics["mae"])
        rmses.append(metrics["rmse"])
    return {"loss": float(np.mean(losses)), "mae": float(np.mean(maes)), "rmse": float(np.mean(rmses))}


def evaluate(model: FrozenFoundationFWI, dataloader: Any, *, device: str = "cpu") -> dict[str, float]:
    torch = require_torch_backend()
    criterion = torch.nn.MSELoss()
    model.eval()
    losses = []
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in dataloader:
            records, velocity, source_positions, _ = _extract_batch(batch, device=device)
            prediction = model(records, source_positions)
            loss = criterion(prediction, velocity)
            losses.append(float(loss.detach().cpu()))
            predictions.append(prediction.detach().cpu().numpy())
            targets.append(velocity.detach().cpu().numpy())
    if not predictions:
        return {"loss": float(np.mean(losses)) if losses else 0.0, "mae": 0.0, "rmse": 0.0}
    metrics = compute_velocity_metrics(
        np.concatenate(predictions, axis=0),
        np.concatenate(targets, axis=0),
        data_range=1.0,
    )
    metrics["loss"] = float(np.mean(losses)) if losses else float(metrics["loss"])
    return metrics


def _write_metrics_json(path: Path, metrics: dict[str, float]) -> None:
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_openfwi_dataloader(dataset: Any, *, batch_size: int, shuffle: bool = False, num_workers: int = 0, seed: int = 0) -> Any:
    torch = require_torch_backend()
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def _build_summary(
    *,
    model: FrozenFoundationFWI,
    history: list[dict[str, float]],
    test_metrics: dict[str, float] | None,
    config: dict[str, Any],
    parameter_counts: dict[str, Any],
    foundation_backbone: str,
    backbone_type: str | None,
    model_name: str | None,
    pretrained: bool,
    freeze_backbone: bool,
    transfer_mode: str | None,
    peft_type: str,
    lora_rank: int,
    lora_alpha: float,
    lora_dropout: float,
    lora_target_modules: tuple[str, ...],
    image_size: int,
    in_chans: int,
    aggregation: str,
    depth: int,
    width: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    device: str,
    sample_count: int,
    split_counts: dict[str, int],
    model_type: str,
    test_evaluated: bool = True,
    test_reason: str | None = None,
    test_split: str | None = None,
) -> dict[str, Any]:
    resolved_backbone_type, resolved_backbone_name = _resolve_backbone_identity(
        foundation_backbone=foundation_backbone,
        backbone_type=backbone_type,
        model_name=model_name,
    )
    if model_type == "foundation_fwi":
        summary_model_type = "foundation_fwi"
    elif resolved_backbone_type == "dummy":
        summary_model_type = "dummy_dinov2_baseline"
    elif resolved_backbone_type == "timm" and not pretrained and not freeze_backbone and peft_type == "none":
        summary_model_type = "vit_from_scratch"
    elif pretrained and freeze_backbone and peft_type == "none":
        summary_model_type = "pretrained_vit_frozen"
    elif peft_type == "adapter":
        summary_model_type = "pretrained_vit_adapter"
    elif peft_type == "lora":
        summary_model_type = "pretrained_vit_lora"
    else:
        summary_model_type = "foundation_transfer_baseline"
    return {
        "model_type": summary_model_type,
        "backbone_name": resolved_backbone_name,
        "backbone_type": resolved_backbone_type,
        "pretrained": bool(pretrained),
        "freeze_backbone": bool(freeze_backbone),
        "transfer_mode": transfer_mode,
        "peft_type": peft_type,
        "lora_rank": int(lora_rank),
        "lora_alpha": float(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "lora_target_modules": list(lora_target_modules),
        "injected_lora_modules": int(getattr(model, "injected_lora_modules", 0)),
        "injected_adapter_modules": int(getattr(model, "injected_adapter_modules", 0)),
        "image_size": int(image_size),
        "in_chans": int(in_chans),
        "aggregation": aggregation,
        "bridge_feature_mode": config.get("bridge_feature_mode", "raw_repeat3"),
        "spectrogram_n_fft": config.get("spectrogram_n_fft", 64),
        "spectrogram_hop_length": config.get("spectrogram_hop_length", 16),
        "spectrogram_win_length": config.get("spectrogram_win_length", 64),
        "spectrogram_power": config.get("spectrogram_power", 1.0),
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
        "sample_count": int(sample_count),
        "split_counts": split_counts,
        "final_train": history[-1],
        "test_metrics": test_metrics or {},
        "test_evaluated": bool(test_evaluated),
        "test_reason": test_reason,
        "test_split": test_split,
        "config": config,
    }


def run_foundation_npz_experiment(
    *,
    data_dir: str | Path | None,
    output_dir: str | Path,
    depth: int,
    width: int,
    foundation_backbone: str = "dummy_dinov2",
    backbone_type: str | None = None,
    model_name: str | None = None,
    pretrained: bool = False,
    freeze_backbone: bool = True,
    peft_type: str = "none",
    lora_rank: int = 4,
    lora_alpha: float = 8.0,
    lora_dropout: float = 0.0,
    lora_target_modules: tuple[str, ...] = ("qkv", "proj", "fc1", "fc2"),
    train_bias: bool = False,
    adapter_bottleneck_dim: int = 64,
    adapter_dropout: float = 0.0,
    image_size: int = 224,
    in_chans: int = 3,
    norm_mode: str = "zscore",
    remove_cls_token: bool = False,
    local_files_only: bool = False,
    model_type: str = "baseline",
    transfer_mode: str | None = None,
    print_parameter_report: bool = True,
    aggregation: str = "mean",
    bridge_feature_mode: str = "raw_repeat3",
    spectrogram_n_fft: int = 64,
    spectrogram_hop_length: int = 16,
    spectrogram_win_length: int = 64,
    spectrogram_power: float = 1.0,
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
) -> dict[str, Any]:
    torch = require_torch_backend()
    set_torch_seed(seed)
    effective_freeze_backbone = bool(freeze_backbone or transfer_mode in {"frozen", "adapter", "lora"})
    if transfer_mode in {"adapter", "lora"} and peft_type == "none":
        peft_type = transfer_mode
    if peft_type in {"adapter", "lora"} and transfer_mode is None:
        transfer_mode = peft_type
    if transfer_mode in {"adapter", "lora"} and peft_type != transfer_mode:
        raise ValueError(f"transfer_mode={transfer_mode} requires matching peft_type={transfer_mode}, got {peft_type}")
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
        "foundation_backbone": foundation_backbone,
        "backbone_type": backbone_type,
        "model_name": model_name,
        "pretrained": bool(pretrained),
        "freeze_backbone": effective_freeze_backbone,
        "peft_type": peft_type,
        "lora_rank": int(lora_rank),
        "lora_alpha": float(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "lora_target_modules": list(lora_target_modules),
        "train_bias": bool(train_bias),
        "adapter_bottleneck_dim": int(adapter_bottleneck_dim),
        "adapter_dropout": float(adapter_dropout),
        "image_size": int(image_size),
        "in_chans": int(in_chans),
        "norm_mode": norm_mode,
        "remove_cls_token": bool(remove_cls_token),
        "local_files_only": bool(local_files_only),
        "model_type": model_type,
        "transfer_mode": transfer_mode,
        "print_parameter_report": bool(print_parameter_report),
        "aggregation": aggregation,
        "bridge_feature_mode": bridge_feature_mode,
        "spectrogram_n_fft": int(spectrogram_n_fft),
        "spectrogram_hop_length": int(spectrogram_hop_length),
        "spectrogram_win_length": int(spectrogram_win_length),
        "spectrogram_power": float(spectrogram_power),
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
    resolved_payload = json.dumps(config, indent=2, ensure_ascii=False)
    (output / "resolved_foundation_config.json").write_text(resolved_payload, encoding="utf-8")
    (output / "config_resolved.json").write_text(resolved_payload, encoding="utf-8")
    train_loader = build_torch_dataloader(split["train"], batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = build_torch_dataloader(split["val"] or split["train"], batch_size=batch_size, shuffle=False, seed=seed)
    test_loader = build_torch_dataloader(split["test"] or split["train"], batch_size=batch_size, shuffle=False, seed=seed)
    model = FrozenFoundationFWI(
        foundation_backbone=foundation_backbone,
        backbone_type=backbone_type,
        model_name=model_name,
        pretrained=pretrained,
        freeze_backbone=effective_freeze_backbone,
        peft_type=peft_type,
        lora_config=LoRAConfig(
            enabled=peft_type == "lora",
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=tuple(lora_target_modules),
            train_bias=train_bias,
        ),
        adapter_config=AdapterConfig(
            enabled=peft_type == "adapter",
            bottleneck_dim=adapter_bottleneck_dim,
            dropout=adapter_dropout,
        ),
        image_size=image_size,
        in_chans=in_chans,
        norm_mode=norm_mode,
        remove_cls_token=remove_cls_token,
        local_files_only=local_files_only,
        depth=depth,
        width=width,
        aggregation=aggregation,
        bridge_feature_mode=bridge_feature_mode,
        spectrogram_n_fft=spectrogram_n_fft,
        spectrogram_hop_length=spectrogram_hop_length,
        spectrogram_win_length=spectrogram_win_length,
        spectrogram_power=spectrogram_power,
        vmin=vmin,
        vmax=vmax,
        device=device,
        transfer_mode=transfer_mode,
        print_parameter_report=print_parameter_report,
    ).to(device)
    total_parameters, trainable_parameters, trainable_ratio = count_parameters(model.module)
    parameter_counts = {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_ratio": trainable_ratio,
    }
    parameter_report = getattr(model.module, "parameter_report", None)
    if parameter_report is not None:
        report_text = str(parameter_report.get("report", ""))
    else:
        report_text = ""
    (output / "parameter_report.txt").write_text(report_text, encoding="utf-8")
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device=device)
        val_metrics = evaluate(model, val_loader, device=device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
        }
        history.append(row)
        save_foundation_checkpoint(output, model, optimizer, config, row)
    test_metrics = evaluate(model, test_loader, device=device)
    _save_prediction_samples(model, test_loader, output, device=device)
    (output / "foundation_training_history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "foundation_training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    summary = _build_summary(
        model=model,
        history=history,
        test_metrics=test_metrics,
        config=config,
        parameter_counts=parameter_counts,
        foundation_backbone=foundation_backbone,
        backbone_type=backbone_type,
        model_name=model_name,
        pretrained=pretrained,
        freeze_backbone=effective_freeze_backbone,
        transfer_mode=transfer_mode,
        peft_type=peft_type,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        image_size=image_size,
        in_chans=in_chans,
        aggregation=aggregation,
        depth=depth,
        width=width,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        sample_count=len(paths),
        split_counts={name: len(items) for name, items in split.items()},
        model_type=model_type,
    )
    (output / "foundation_experiment_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def run_foundation_openfwi_experiment(
    *,
    openfwi_root: str | Path,
    manifest_path: str | Path,
    train_split: str | Path,
    val_split: str | Path,
    test_split: str | Path | None,
    stats_file: str | Path,
    output_dir: str | Path,
    foundation_backbone: str = "dummy_dinov2",
    backbone_type: str | None = None,
    model_name: str | None = None,
    pretrained: bool = False,
    freeze_backbone: bool = True,
    peft_type: str = "none",
    lora_rank: int = 4,
    lora_alpha: float = 8.0,
    lora_dropout: float = 0.0,
    lora_target_modules: tuple[str, ...] = ("qkv", "proj", "fc1", "fc2"),
    train_bias: bool = False,
    adapter_bottleneck_dim: int = 64,
    adapter_dropout: float = 0.0,
    image_size: int = 224,
    in_chans: int = 3,
    norm_mode: str = "zscore",
    remove_cls_token: bool = False,
    local_files_only: bool = False,
    model_type: str = "foundation_fwi",
    transfer_mode: str | None = None,
    print_parameter_report: bool = True,
    aggregation: str = "mean",
    bridge_feature_mode: str = "raw_repeat3",
    spectrogram_n_fft: int = 64,
    spectrogram_hop_length: int = 16,
    spectrogram_win_length: int = 64,
    spectrogram_power: float = 1.0,
    batch_size: int = 4,
    epochs: int = 5,
    learning_rate: float = 1.0e-3,
    vmin: float = 1500.0,
    vmax: float = 4500.0,
    device: str = "cpu",
    seed: int = 0,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
    input_norm: str = "zscore",
    target_norm: str = "minmax",
    save_predictions: bool = False,
    num_preview: int = 8,
) -> dict[str, Any]:
    torch = require_torch_backend()
    set_torch_seed(seed)
    effective_freeze_backbone = bool(freeze_backbone or transfer_mode in {"frozen", "adapter", "lora"})
    if transfer_mode in {"adapter", "lora"} and peft_type == "none":
        peft_type = transfer_mode
    if peft_type in {"adapter", "lora"} and transfer_mode is None:
        transfer_mode = peft_type
    train_dataset = OpenFWINpyDataset(
        root=str(openfwi_root),
        split_file=str(train_split),
        stats_file=str(stats_file),
        fit_stats=False,
        input_norm=input_norm,
        target_norm=target_norm,
        max_samples=max_train_samples,
    )
    val_dataset = OpenFWINpyDataset(
        root=str(openfwi_root),
        split_file=str(val_split),
        stats_file=str(stats_file),
        fit_stats=False,
        input_norm=input_norm,
        target_norm=target_norm,
        max_samples=max_val_samples,
    )
    test_dataset = None
    if test_split is not None:
        test_dataset = OpenFWINpyDataset(
            root=str(openfwi_root),
            split_file=str(test_split),
            stats_file=str(stats_file),
            fit_stats=False,
            input_norm=input_norm,
            target_norm=target_norm,
            max_samples=max_test_samples,
        )
    first_sample = train_dataset[0]
    records_shape = tuple(int(value) for value in first_sample["seismic"].shape)
    velocity_shape = tuple(int(value) for value in first_sample["velocity"].shape[-2:])
    depth, width = velocity_shape

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "openfwi_root": str(Path(openfwi_root)),
        "manifest": str(Path(manifest_path)),
        "train_split": str(Path(train_split)),
        "val_split": str(Path(val_split)),
        "test_split": "" if test_split is None else str(Path(test_split)),
        "stats_file": str(Path(stats_file)),
        "depth": int(depth),
        "width": int(width),
        "records_shape": list(records_shape),
        "foundation_backbone": foundation_backbone,
        "backbone_type": backbone_type,
        "model_name": model_name,
        "pretrained": bool(pretrained),
        "freeze_backbone": effective_freeze_backbone,
        "peft_type": peft_type,
        "lora_rank": int(lora_rank),
        "lora_alpha": float(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "lora_target_modules": list(lora_target_modules),
        "train_bias": bool(train_bias),
        "adapter_bottleneck_dim": int(adapter_bottleneck_dim),
        "adapter_dropout": float(adapter_dropout),
        "image_size": int(image_size),
        "in_chans": int(in_chans),
        "norm_mode": norm_mode,
        "input_norm": input_norm,
        "target_norm": target_norm,
        "remove_cls_token": bool(remove_cls_token),
        "local_files_only": bool(local_files_only),
        "model_type": model_type,
        "transfer_mode": transfer_mode,
        "print_parameter_report": bool(print_parameter_report),
        "aggregation": aggregation,
        "bridge_feature_mode": bridge_feature_mode,
        "spectrogram_n_fft": int(spectrogram_n_fft),
        "spectrogram_hop_length": int(spectrogram_hop_length),
        "spectrogram_win_length": int(spectrogram_win_length),
        "spectrogram_power": float(spectrogram_power),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "vmin": float(vmin),
        "vmax": float(vmax),
        "device": device,
        "seed": int(seed),
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
        "max_test_samples": max_test_samples,
        "save_predictions": bool(save_predictions),
        "num_preview": int(num_preview),
    }
    resolved_payload = json.dumps(config, indent=2, ensure_ascii=False)
    (output / "resolved_foundation_config.json").write_text(resolved_payload, encoding="utf-8")
    (output / "config_resolved.json").write_text(resolved_payload, encoding="utf-8")

    train_loader = _build_openfwi_dataloader(train_dataset, batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = _build_openfwi_dataloader(val_dataset, batch_size=batch_size, shuffle=False, seed=seed)
    test_loader = None if test_dataset is None else _build_openfwi_dataloader(test_dataset, batch_size=batch_size, shuffle=False, seed=seed)
    model = FrozenFoundationFWI(
        foundation_backbone=foundation_backbone,
        backbone_type=backbone_type,
        model_name=model_name,
        pretrained=pretrained,
        freeze_backbone=effective_freeze_backbone,
        peft_type=peft_type,
        lora_config=LoRAConfig(
            enabled=peft_type == "lora",
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=tuple(lora_target_modules),
            train_bias=train_bias,
        ),
        adapter_config=AdapterConfig(
            enabled=peft_type == "adapter",
            bottleneck_dim=adapter_bottleneck_dim,
            dropout=adapter_dropout,
        ),
        image_size=image_size,
        in_chans=in_chans,
        norm_mode=norm_mode,
        remove_cls_token=remove_cls_token,
        local_files_only=local_files_only,
        depth=depth,
        width=width,
        aggregation=aggregation,
        bridge_feature_mode=bridge_feature_mode,
        spectrogram_n_fft=spectrogram_n_fft,
        spectrogram_hop_length=spectrogram_hop_length,
        spectrogram_win_length=spectrogram_win_length,
        spectrogram_power=spectrogram_power,
        vmin=vmin,
        vmax=vmax,
        device=device,
        transfer_mode=transfer_mode,
        print_parameter_report=print_parameter_report,
    ).to(device)
    total_parameters, trainable_parameters, trainable_ratio = count_parameters(model.module)
    parameter_counts = {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_ratio": trainable_ratio,
    }
    parameter_report = getattr(model.module, "parameter_report", None)
    report_text = str(parameter_report.get("report", "")) if parameter_report is not None else ""
    (output / "parameter_report.txt").write_text(report_text, encoding="utf-8")
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device=device)
        val_metrics = evaluate(model, val_loader, device=device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
        }
        history.append(row)
        save_foundation_checkpoint(output, model, optimizer, config, row)
    val_metrics = evaluate(model, val_loader, device=device)
    test_metrics = evaluate(model, test_loader, device=device) if test_loader is not None else None
    _write_metrics_json(output / "metrics_val.json", val_metrics)
    if test_metrics is not None:
        _write_metrics_json(output / "metrics_test.json", test_metrics)
        _save_prediction_samples(model, test_loader, output, device=device)
    if save_predictions:
        preview_loader = test_loader if test_loader is not None else val_loader
        _save_predictions_preview(model, preview_loader, output / "predictions_preview.npz", device=device, limit=num_preview)
    (output / "foundation_training_history.json").write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output / "foundation_training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    with (output / "training_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    summary = _build_summary(
        model=model,
        history=history,
        test_metrics=test_metrics,
        config=config,
        parameter_counts=parameter_counts,
        foundation_backbone=foundation_backbone,
        backbone_type=backbone_type,
        model_name=model_name,
        pretrained=pretrained,
        freeze_backbone=effective_freeze_backbone,
        transfer_mode=transfer_mode,
        peft_type=peft_type,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=lora_target_modules,
        image_size=image_size,
        in_chans=in_chans,
        aggregation=aggregation,
        depth=depth,
        width=width,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        sample_count=len(train_dataset) + len(val_dataset) + (0 if test_dataset is None else len(test_dataset)),
        split_counts={"train": len(train_dataset), "val": len(val_dataset), "test": 0 if test_dataset is None else len(test_dataset)},
        model_type=model_type,
        test_evaluated=test_dataset is not None,
        test_reason=None if test_dataset is not None else "no test split provided",
        test_split=None if test_split is None else str(Path(test_split)),
    )
    (output / "foundation_experiment_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def generate_foundation_synthetic_npz(
    output_dir: str | Path,
    *,
    samples: int = 4,
    shots: int = 3,
    receivers: int = 8,
    time_samples: int = 12,
    depth: int = 6,
    width: int = 7,
    seed: int = 0,
) -> Path:
    output_path = Path(output_dir)
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
        output_path,
        dataset_name="foundation_synthetic_smoke",
        source_positions=np.stack(source_positions, axis=0),
    )
    return output_path
