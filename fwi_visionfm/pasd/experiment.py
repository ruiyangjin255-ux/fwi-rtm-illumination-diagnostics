"""Reusable PASD training/evaluation core shared by single runs and protocol matrices."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import ArrayBundle, OpenFWINpyDataset, VelocityScaler
from .losses import BackgroundEdgeLoss, LossOutput, VelocityCriterion, VelocityL1Loss
from .metrics import per_sample_metrics
from .model import PASDFWI
from .plotting import plot_bridge_attributes, plot_bridge_attribute_files, plot_gradient_comparison, plot_profiles, plot_training_history, plot_velocity_comparison
from .registry import PASDVariant, get_variant


@dataclass(frozen=True)
class TrainingConfig:
    """All training hyperparameters that must be archived for reproducibility."""

    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cpu"
    num_workers: int = 0
    base_channels: int = 16
    latent_channels: int = 96
    latent_size: tuple[int, int] = (9, 9)
    lowpass_kernel: int = 21
    residual_scale: float = 0.25
    background_sigma: float = 1.5
    edge_quantile: float = 0.8
    weight_background: float = 0.25
    weight_edge: float = 0.10
    weight_smooth: float = 0.02
    generate_figures: bool = True


@dataclass(frozen=True)
class ProtocolSplits:
    train: tuple[int, ...]
    val: tuple[int, ...]
    in_family_test: tuple[int, ...]
    cross_family_test: tuple[int, ...] = ()


@dataclass
class RunResult:
    output_dir: str
    variant: str
    seed: int
    in_family: dict[str, float]
    cross_family: dict[str, float] | None
    scaler: VelocityScaler


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(variant: PASDVariant, output_size: tuple[int, int], config: TrainingConfig) -> PASDFWI:
    from .bridge import HybridAttributeBridge

    bridge = HybridAttributeBridge(lowpass_kernel=config.lowpass_kernel)
    return PASDFWI(
        output_size=output_size,
        base_channels=config.base_channels,
        latent_channels=config.latent_channels,
        latent_size=config.latent_size,
        aggregator=variant.aggregator,
        bridge_mode=variant.bridge_mode,
        decoder_mode=variant.decoder_mode,
        residual_scale=config.residual_scale,
        bridge=bridge,
    )


def build_criterion(variant: PASDVariant, config: TrainingConfig) -> VelocityCriterion:
    if variant.criterion == "l1":
        return VelocityL1Loss()
    return BackgroundEdgeLoss(
        background_sigma=config.background_sigma,
        edge_quantile=config.edge_quantile,
        weight_l1=1.0,
        weight_background=config.weight_background,
        weight_edge=config.weight_edge,
        weight_smooth=config.weight_smooth,
    )


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / max(1, len(values)))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _make_loader(bundle: ArrayBundle, indices: tuple[int, ...], scaler: VelocityScaler, config: TrainingConfig, shuffle: bool) -> DataLoader:
    dataset = OpenFWINpyDataset(
        bundle.records, bundle.velocities, indices, scaler=scaler, sample_ids=bundle.sample_ids,
        source_positions=bundle.source_positions, receiver_positions=bundle.receiver_positions
    )
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_workers)


def _physical_metrics(prediction_normalized: Tensor, target_normalized: Tensor, scaler: VelocityScaler) -> dict[str, Tensor]:
    scale = float(scaler.maximum - scaler.minimum)
    prediction = prediction_normalized * scale + scaler.minimum
    target = target_normalized * scale + scaler.minimum
    return per_sample_metrics(prediction, target)


def evaluate_model(
    model: PASDFWI,
    loader: DataLoader,
    device: torch.device,
    scaler: VelocityScaler,
    criterion: VelocityCriterion | None = None,
) -> tuple[dict[str, float], list[dict[str, float]], dict[str, np.ndarray] | None, dict[str, np.ndarray]]:
    """Evaluate one split and retain per-sample archives for aligned bootstrap tests."""

    model.eval()
    metric_rows: list[dict[str, float]] = []
    cached: dict[str, np.ndarray] | None = None
    losses: list[float] = []
    archive_ids: list[np.ndarray] = []
    archive_predictions: list[np.ndarray] = []
    archive_targets: list[np.ndarray] = []
    archive_attention: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            records = batch["records"].to(device)
            velocity = batch["velocity"].to(device)
            source_positions = batch.get("source_positions")
            receiver_positions = batch.get("receiver_positions")
            if source_positions is not None:
                source_positions = source_positions.to(device)
            if receiver_positions is not None:
                receiver_positions = receiver_positions.to(device)
            output = model(records, source_positions=source_positions, receiver_positions=receiver_positions)
            if criterion is not None:
                losses.append(float(criterion(output.velocity, output.background, velocity).total.item()))
            metrics = _physical_metrics(output.velocity, velocity, scaler)
            sample_ids = batch["sample_id"].cpu().numpy().tolist()
            for row_index, sample_id in enumerate(sample_ids):
                row: dict[str, float] = {"sample_id": int(sample_id)}
                for name, values in metrics.items():
                    row[name] = float(values[row_index].cpu().item())
                metric_rows.append(row)
            if cached is None:
                cached = {
                    "records": records[:1].cpu().numpy(),
                    "truth": velocity[:1].cpu().numpy(),
                    "prediction": output.velocity[:1].cpu().numpy(),
                    "bridge": output.bridge.attributes[:1].cpu().numpy(),
                    "attention": output.attention[:1].cpu().numpy(),
                }
            archive_ids.append(batch["sample_id"].cpu().numpy())
            archive_predictions.append(output.velocity[:, 0].cpu().numpy())
            archive_targets.append(velocity[:, 0].cpu().numpy())
            archive_attention.append(output.attention.cpu().numpy())

    metrics_to_mean = ["mae", "rmse", "ssim", "psnr", "edge_mae", "gradient_error", "laplacian_error", "relative_error"]
    summary = {name: _mean(float(row[name]) for row in metric_rows) for name in metrics_to_mean}
    if losses:
        summary["loss"] = _mean(losses)
    archive = {
        "sample_id": np.concatenate(archive_ids),
        "prediction": np.concatenate(archive_predictions),
        "target": np.concatenate(archive_targets),
        "attention": np.concatenate(archive_attention),
    }
    return summary, metric_rows, cached, archive


def _save_archive(path: Path, archive: dict[str, np.ndarray], scaler: VelocityScaler) -> None:
    prediction = scaler.denormalize(archive["prediction"])
    target = scaler.denormalize(archive["target"])
    np.savez_compressed(
        path,
        sample_id=archive["sample_id"],
        prediction=prediction,
        target=target,
        attention=archive["attention"],
    )


def run_single_experiment(
    source: ArrayBundle,
    splits: ProtocolSplits,
    output_dir: str | Path,
    variant_name: str,
    config: TrainingConfig,
    target: ArrayBundle | None = None,
) -> RunResult:
    """Train solely on source.train and evaluate in-family and target-family splits.

    The velocity scaler is fit only on source.train. Target data are instantiated after the scaler
    is frozen, and no target labels or records are visible to training/validation.
    """

    variant = get_variant(variant_name)
    seed_everything(config.seed)
    output_dir = Path(output_dir)
    figures = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)

    train_velocities = source.velocities[np.asarray(splits.train, dtype=np.int64)]
    scaler = VelocityScaler.fit(train_velocities)
    train_loader = _make_loader(source, splits.train, scaler, config, shuffle=True)
    val_loader = _make_loader(source, splits.val, scaler, config, shuffle=False)
    in_loader = _make_loader(source, splits.in_family_test, scaler, config, shuffle=False)

    cross_loader: DataLoader | None = None
    if target is not None:
        if not splits.cross_family_test:
            raise ValueError("Target data were provided but cross_family_test indices are missing.")
        cross_loader = _make_loader(target, splits.cross_family_test, scaler, config, shuffle=False)
    elif splits.cross_family_test:
        raise ValueError("cross_family_test indices require a target bundle.")

    device = torch.device(config.device)
    model = build_model(variant, tuple(int(x) for x in source.velocities.shape[-2:]), config).to(device)
    criterion = build_criterion(variant, config)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        batch_totals: list[float] = []
        component_values: dict[str, list[float]] = {"l1": [], "background": [], "edge": [], "smooth": []}
        for batch in train_loader:
            records = batch["records"].to(device)
            velocity = batch["velocity"].to(device)
            optimizer.zero_grad(set_to_none=True)
            source_positions = batch.get("source_positions")
            receiver_positions = batch.get("receiver_positions")
            if source_positions is not None:
                source_positions = source_positions.to(device)
            if receiver_positions is not None:
                receiver_positions = receiver_positions.to(device)
            output = model(records, source_positions=source_positions, receiver_positions=receiver_positions)
            loss_output = criterion(output.velocity, output.background, velocity)
            loss_output.total.backward()
            optimizer.step()
            batch_totals.append(float(loss_output.total.item()))
            for name, value in loss_output.components.items():
                component_values[name].append(float(value.item()))
        val_summary, _, _, _ = evaluate_model(model, val_loader, device, scaler, criterion)
        row: dict[str, float] = {
            "epoch": epoch,
            "train_loss": _mean(batch_totals),
            "val_loss": float(val_summary.get("loss", float("nan"))),
            **{f"train_{name}": _mean(values) for name, values in component_values.items()},
            **{f"val_{name}": value for name, value in val_summary.items() if name != "loss"},
        }
        history.append(row)
        print(json.dumps({"variant": variant_name, "seed": config.seed, **row}, ensure_ascii=False))

    val_summary, val_rows, _, val_archive = evaluate_model(model, val_loader, device, scaler, criterion)
    in_summary, in_rows, in_cached, in_archive = evaluate_model(model, in_loader, device, scaler, criterion)
    cross_summary: dict[str, float] | None = None
    cross_rows: list[dict[str, float]] = []
    cross_archive: dict[str, np.ndarray] | None = None
    if cross_loader is not None:
        cross_summary, cross_rows, _, cross_archive = evaluate_model(model, cross_loader, device, scaler, criterion)

    _write_csv(output_dir / "history.csv", history)
    _write_csv(output_dir / "metrics_val_per_sample.csv", val_rows)
    _write_csv(output_dir / "metrics_in_family_per_sample.csv", in_rows)
    _save_archive(output_dir / "predictions_val.npz", val_archive, scaler)
    _save_archive(output_dir / "predictions_in_family.npz", in_archive, scaler)
    if cross_summary is not None and cross_archive is not None:
        _write_csv(output_dir / "metrics_cross_family_per_sample.csv", cross_rows)
        _save_archive(output_dir / "predictions_cross_family.npz", cross_archive, scaler)

    metadata = {
        "status": "SUCCESS",
        "variant": variant_name,
        "variant_description": variant.description,
        "seed": config.seed,
        "source_family": source.family,
        "target_family": target.family if target is not None else None,
        "splits": {
            "train_size": len(splits.train),
            "val_size": len(splits.val),
            "in_family_test_size": len(splits.in_family_test),
            "cross_family_test_size": len(splits.cross_family_test),
        },
        "train_only_scaler": scaler.as_dict(),
        "target_isolation": "Target-family samples were not used for scaler fitting, training, validation, checkpoint selection, or hyperparameter selection.",
        "training": asdict(config),
        "metric_space": "physical_velocity_after_train_only_inverse_transform",
        "velocity_output_constraint": "normalized sigmoid/clamp output denormalized by source-train velocity scaler",
        "metrics": {"val": val_summary, "in_family": in_summary, "cross_family": cross_summary},
    }
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    if config.generate_figures and in_cached is not None:
        truth = scaler.denormalize(in_cached["truth"][0, 0])
        prediction = scaler.denormalize(in_cached["prediction"][0, 0])
        plot_bridge_attributes(in_cached["bridge"], figures / "bridge_preview.png")
        plot_bridge_attribute_files(in_cached["bridge"], figures / "bridge_attributes")
        plot_velocity_comparison(truth, {variant_name: prediction}, figures / "in_family_velocity_comparison.png")
        plot_profiles(truth, {variant_name: prediction}, figures / "in_family_velocity_profiles.png")
        plot_gradient_comparison(truth, {variant_name: prediction}, figures / "in_family_gradient_comparison.png")
        np.save(output_dir / "attention_first_in_family_sample.npy", in_cached["attention"])
    if config.generate_figures:
        plot_training_history(history, figures / "training_history.png", title=f"{variant_name}, seed={config.seed}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "variant": variant_name,
            "training": asdict(config),
            "scaler": scaler.as_dict(),
        },
        output_dir / "checkpoint.pt",
    )
    return RunResult(
        output_dir=str(output_dir),
        variant=variant_name,
        seed=config.seed,
        in_family=in_summary,
        cross_family=cross_summary,
        scaler=scaler,
    )
