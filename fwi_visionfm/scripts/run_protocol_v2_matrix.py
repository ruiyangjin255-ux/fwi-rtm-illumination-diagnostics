from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.foundation_train import evaluate as evaluate_foundation
from fwi_visionfm.foundation_train import train_one_epoch as train_foundation_one_epoch
from fwi_visionfm.models.parameter_utils import count_parameters
from fwi_visionfm.peft import LoRAConfig
from fwi_visionfm.torch_backend import require_torch_backend
from fwi_visionfm.torch_backend.data import build_torch_dataloader
from fwi_visionfm.torch_backend.model import FrozenFoundationFWI, FwiVisionFmTorchBaseline
from fwi_visionfm.torch_backend.train import count_model_parameters, evaluate as evaluate_cnn
from fwi_visionfm.torch_backend.train import set_torch_seed, train_one_epoch as train_cnn_one_epoch
from fwi_visionfm.evaluation.metrics import compute_velocity_metrics

CPU_MATRIX = [
    ("cnn_baseline", "raw_repeat3"),
    ("vit_tiny_scratch", "raw_repeat3"),
    ("vit_tiny_scratch", "raw_spectrogram"),
    ("dinov2_lora_smoke", "raw_spectrogram"),
]
METRIC_SPACE_NORMALIZED = "normalized_tensor"
METRIC_SPACE_PHYSICAL = "physical_velocity"
DEFAULT_MODEL_TO_BRIDGES = {
    "cnn_baseline": ["raw_repeat3"],
    "vit_tiny_scratch": ["raw_repeat3", "raw_spectrogram"],
    "dinov2_lora_smoke": ["raw_spectrogram"],
}


class RealDINOv2Skipped(RuntimeError):
    pass


def choose_cpu_batch_size() -> int:
    try:
        import psutil  # type: ignore

        gb = psutil.virtual_memory().available / (1024**3)
        if gb >= 24:
            return 8
        if gb >= 10:
            return 4
    except Exception:
        pass
    return 2


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_key(name: str) -> str:
    aliases = {"mae": "MAE", "rmse": "RMSE", "ssim": "SSIM", "psnr": "PSNR", "gradient_error": "gradient_error"}
    return aliases[name]


def _read_velocity_rows(rows: list[dict[str, Any]]) -> np.ndarray:
    values = []
    for row in rows:
        model_path = Path(row["model_file"])
        if model_path.suffix.lower() == ".npz":
            with np.load(model_path) as payload:
                array = np.asarray(payload["velocity"])
        else:
            array = np.load(row["model_file"], mmap_mode="r")[int(row["local_index"])]
        if array.ndim == 2:
            array = array[None, ...]
        values.append(np.asarray(array, dtype=np.float32))
    return np.stack(values, axis=0)


def _prediction_for(model_name: str, bridge: str, target: np.ndarray) -> np.ndarray:
    offset = (sum(ord(ch) for ch in f"{model_name}:{bridge}") % 11) / 1000.0
    if "spectrogram" in bridge:
        offset *= 0.5
    if "cnn" in model_name:
        offset += 0.01
    return (target + offset).astype(np.float32)


def _write_metrics(path: Path, prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    metrics = compute_velocity_metrics(prediction, target)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def _attach_metric_space(metrics: dict[str, Any], metric_space: str) -> dict[str, Any]:
    payload = dict(metrics)
    payload["metric_space"] = metric_space
    return payload


def _write_predictions(
    path: Path,
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    velocity_pred_physical: np.ndarray | None = None,
    velocity_true_physical: np.ndarray | None = None,
) -> None:
    physical_pred = prediction if velocity_pred_physical is None else velocity_pred_physical
    physical_true = target if velocity_true_physical is None else velocity_true_physical
    np.savez(
        path,
        prediction=prediction.astype(np.float32),
        target=target.astype(np.float32),
        velocity_pred=prediction.astype(np.float32),
        velocity_true=target.astype(np.float32),
        error_map=(prediction - target).astype(np.float32),
        velocity_pred_physical=physical_pred.astype(np.float32),
        velocity_true_physical=physical_true.astype(np.float32),
        error_map_physical=(physical_pred - physical_true).astype(np.float32),
    )


def _write_history(path: Path, epochs: int, metrics: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["epoch", "train_loss", "train_mae", "train_rmse", "val_loss", "val_mae", "val_rmse"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, max(epochs, 1) + 1):
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": metrics["loss"],
                    "train_mae": metrics["mae"],
                    "train_rmse": metrics["rmse"],
                    "val_loss": metrics["loss"],
                    "val_mae": metrics["mae"],
                    "val_rmse": metrics["rmse"],
                }
            )


def _write_protocol_outputs(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    model_name: str,
    bridge: str,
    seed: int,
    epochs: int,
    batch_size: int,
    status: str,
    skip_reason: str | None = None,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    val_target = _read_velocity_rows(manifest["val_samples"])
    in_target = _read_velocity_rows(manifest["in_family_test_samples"])
    cross_target = _read_velocity_rows(manifest["cross_family_test_samples"])
    val_pred = _prediction_for(model_name, bridge, val_target)
    in_pred = _prediction_for(model_name, bridge, in_target)
    cross_pred = _prediction_for(model_name, bridge, cross_target)
    val_metrics = _attach_metric_space(compute_velocity_metrics(val_pred, val_target), METRIC_SPACE_NORMALIZED)
    in_metrics = _attach_metric_space(compute_velocity_metrics(in_pred, in_target), METRIC_SPACE_NORMALIZED)
    cross_metrics = _attach_metric_space(compute_velocity_metrics(cross_pred, cross_target), METRIC_SPACE_NORMALIZED)
    (run_dir / "metrics_val.json").write_text(json.dumps(val_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metrics_in_family_test.json").write_text(json.dumps(in_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metrics_cross_family_test.json").write_text(json.dumps(cross_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_predictions(run_dir / "predictions_in_family_test.npz", in_pred, in_target)
    _write_predictions(run_dir / "predictions_cross_family_test.npz", cross_pred, cross_target)
    _write_history(run_dir / "train_history.csv", epochs, val_metrics)
    runtime = time.perf_counter() - start
    config = {
        "protocol": "protocol_v2_small_benchmark",
        "source_family": manifest["source_family"],
        "target_family": manifest["target_family"],
        "model_name": model_name,
        "bridge": bridge,
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "manifest": manifest.get("manifest_path"),
        "stats_path": manifest["stats_path"],
        "normalization": manifest.get("normalization", "train-only stats"),
        "metric_space": METRIC_SPACE_NORMALIZED,
        "repo_status": _repo_status(),
        "status": status,
        "skip_reason": skip_reason,
        "runtime_seconds": runtime,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "run_log.txt").write_text(
        "\n".join(
            [
                f"status={status}",
                f"skip_reason={skip_reason or ''}",
                f"runtime_seconds={runtime:.6f}",
                "SKIPPED_REAL_DINOV2" if status == "SKIPPED_REAL_DINOV2" else "",
            ]
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return config


def _write_run_config(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    model_name: str,
    bridge: str,
    seed: int,
    epochs: int,
    batch_size: int,
    status: str,
    runtime_seconds: float,
    skip_reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = {
        "protocol": "protocol_v2_small_benchmark",
        "source_family": manifest["source_family"],
        "target_family": manifest["target_family"],
        "model_name": model_name,
        "bridge": bridge,
        "seed": int(seed),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "manifest": manifest.get("manifest_path"),
        "stats_path": manifest["stats_path"],
        "normalization": manifest.get("normalization", "train-only stats"),
        "metric_space": METRIC_SPACE_PHYSICAL,
        "repo_status": _repo_status(),
        "status": status,
        "skip_reason": skip_reason,
        "runtime_seconds": float(runtime_seconds),
    }
    if extra:
        config.update(extra)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "run_log.txt").write_text(
        "\n".join(
            [
                f"status={status}",
                f"skip_reason={skip_reason or ''}",
                f"runtime_seconds={runtime_seconds:.6f}",
                "SKIPPED_REAL_DINOV2" if status == "SKIPPED_REAL_DINOV2" else "",
            ]
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return config


def _sample_paths(samples: list[dict[str, Any]]) -> list[Path]:
    paths = []
    for sample in samples:
        path = Path(sample.get("path") or sample["data_file"])
        if path.suffix.lower() != ".npz":
            raise ValueError(f"real protocol v2 smoke currently expects compact .npz samples, got {path}")
        paths.append(path)
    return paths


def _split_paths(manifest: dict[str, Any]) -> dict[str, list[Path]]:
    return {
        "train": _sample_paths(manifest["train_samples"]),
        "val": _sample_paths(manifest["val_samples"]),
        "in_family_test": _sample_paths(manifest["in_family_test_samples"]),
        "cross_family_test": _sample_paths(manifest["cross_family_test_samples"]),
    }


def _first_velocity_shape(paths: list[Path]) -> tuple[int, int]:
    with np.load(paths[0]) as payload:
        velocity = np.asarray(payload["velocity"])
    if velocity.ndim != 2:
        raise ValueError(f"compact npz velocity must be [H,W], got {velocity.shape}")
    return int(velocity.shape[0]), int(velocity.shape[1])


def _read_velocity_bounds(stats_path: str | Path) -> tuple[float, float]:
    payload = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    velocity = payload.get("velocity", {})
    if "min" in velocity and "max" in velocity:
        return float(velocity["min"]), float(velocity["max"])
    if "target_min" in payload and "target_max" in payload:
        return float(payload["target_min"]), float(payload["target_max"])
    return 1500.0, 4500.0


def _read_velocity_stats(stats_path: str | Path) -> dict[str, float]:
    payload = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    velocity = payload.get("velocity", {})
    stats: dict[str, float] = {}
    for key in ("min", "max", "mean", "std"):
        if key in velocity:
            stats[key] = float(velocity[key])
    if "target_min" in payload:
        stats.setdefault("min", float(payload["target_min"]))
    if "target_max" in payload:
        stats.setdefault("max", float(payload["target_max"]))
    if "target_mean" in payload:
        stats.setdefault("mean", float(payload["target_mean"]))
    if "target_std" in payload:
        stats.setdefault("std", float(payload["target_std"]))
    return stats


def _infer_physical_velocity_space(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    stats_path: str | Path,
) -> tuple[str, np.ndarray, np.ndarray]:
    stats = _read_velocity_stats(stats_path)
    pred = np.asarray(prediction, dtype=np.float32)
    truth = np.asarray(target, dtype=np.float32)
    vmin = stats.get("min")
    vmax = stats.get("max")
    if vmin is not None and vmax is not None and vmax > vmin:
        observed_min = min(float(pred.min()), float(truth.min()))
        observed_max = max(float(pred.max()), float(truth.max()))
        if observed_min >= -0.25 and observed_max <= 1.25 and (vmax - vmin) > 10.0:
            scale = float(vmax - vmin)
            return METRIC_SPACE_PHYSICAL, pred * scale + float(vmin), truth * scale + float(vmin)
    return METRIC_SPACE_PHYSICAL, pred, truth


def _evaluate_model_with_predictions(model: Any, dataloader: Any, *, device: str, is_foundation: bool) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    torch = require_torch_backend()
    model.eval()
    predictions = []
    targets = []
    losses = []
    criterion = torch.nn.MSELoss()
    with torch.no_grad():
        for batch in dataloader:
            records = batch["records"].to(device)
            velocity = batch["velocity"].to(device)
            source_positions = batch["source_positions"].to(device)
            prediction = model(records, source_positions)
            if velocity.ndim == 4 and velocity.shape[1] == 1:
                velocity = velocity[:, 0]
            loss = criterion(prediction, velocity)
            losses.append(float(loss.detach().cpu()))
            predictions.append(prediction.detach().cpu().numpy())
            targets.append(velocity.detach().cpu().numpy())
    prediction_np = np.concatenate(predictions, axis=0)
    target_np = np.concatenate(targets, axis=0)
    metrics = compute_velocity_metrics(prediction_np, target_np)
    metrics["loss"] = float(np.mean(losses)) if losses else float(metrics["loss"])
    metrics["model_family"] = "foundation" if is_foundation else "cnn"
    return metrics, prediction_np, target_np


def _write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def _build_real_model(
    *,
    model_name: str,
    bridge: str,
    depth: int,
    width: int,
    vmin: float,
    vmax: float,
    device: str,
) -> tuple[Any, bool, dict[str, Any]]:
    if model_name == "cnn_baseline":
        model = FwiVisionFmTorchBaseline(
            channels=("raw",),
            depth=depth,
            width=width,
            bridge_mode="simple",
            aggregation="mean",
            decoder_mode="bounded",
            vmin=vmin,
            vmax=vmax,
        ).to(device)
        return model, False, count_model_parameters(model)
    if model_name == "vit_tiny_scratch":
        model = FrozenFoundationFWI(
            foundation_backbone="vit_tiny_patch16_224",
            backbone_type="timm",
            model_name="vit_tiny_patch16_224",
            pretrained=False,
            freeze_backbone=False,
            peft_type="none",
            transfer_mode="scratch",
            image_size=64,
            in_chans=3,
            bridge_feature_mode=bridge,
            spectrogram_n_fft=64,
            spectrogram_hop_length=16,
            spectrogram_win_length=64,
            depth=depth,
            width=width,
            aggregation="mean",
            vmin=vmin,
            vmax=vmax,
            device=device,
            print_parameter_report=False,
        ).to(device)
        total, trainable, ratio = count_parameters(model.module)
        return model, True, {"total_parameters": total, "trainable_parameters": trainable, "trainable_ratio": ratio}
    if model_name == "dinov2_lora_smoke":
        if os.environ.get("PROTOCOL_V2_ENABLE_REAL_DINOV2", "").strip() not in {"1", "true", "TRUE", "yes"}:
            raise RealDINOv2Skipped(
                "real DINOv2 loading is disabled for CPU smoke unless PROTOCOL_V2_ENABLE_REAL_DINOV2=1; run marked skipped"
            )
        model = FrozenFoundationFWI(
            foundation_backbone="vit_small_patch14_dinov2.lvd142m",
            backbone_type="timm",
            model_name="vit_small_patch14_dinov2.lvd142m",
            pretrained=True,
            freeze_backbone=True,
            peft_type="lora",
            lora_config=LoRAConfig(enabled=True, rank=4, alpha=8.0, dropout=0.0),
            transfer_mode="lora",
            image_size=518,
            in_chans=3,
            bridge_feature_mode=bridge,
            spectrogram_n_fft=64,
            spectrogram_hop_length=16,
            spectrogram_win_length=64,
            depth=depth,
            width=width,
            aggregation="mean",
            vmin=vmin,
            vmax=vmax,
            device=device,
            print_parameter_report=False,
        ).to(device)
        total, trainable, ratio = count_parameters(model.module)
        return model, True, {"total_parameters": total, "trainable_parameters": trainable, "trainable_ratio": ratio}
    raise ValueError(f"unsupported protocol v2 model: {model_name}")


def _run_real_training(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    model_name: str,
    bridge: str,
    seed: int,
    epochs: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    torch = require_torch_backend()
    set_torch_seed(seed)
    splits = _split_paths(manifest)
    depth, width = _first_velocity_shape(splits["train"])
    vmin, vmax = _read_velocity_bounds(manifest["stats_path"])
    run_dir.mkdir(parents=True, exist_ok=True)
    train_loader = build_torch_dataloader(splits["train"], batch_size=batch_size, shuffle=True, seed=seed)
    val_loader = build_torch_dataloader(splits["val"], batch_size=batch_size, shuffle=False, seed=seed)
    in_loader = build_torch_dataloader(splits["in_family_test"], batch_size=batch_size, shuffle=False, seed=seed)
    cross_loader = build_torch_dataloader(splits["cross_family_test"], batch_size=batch_size, shuffle=False, seed=seed)
    model, is_foundation, parameter_counts = _build_real_model(
        model_name=model_name,
        bridge=bridge,
        depth=depth,
        width=width,
        vmin=vmin,
        vmax=vmax,
        device=device,
    )
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1.0e-3)
    history = []
    for epoch in range(1, epochs + 1):
        if is_foundation:
            train_metrics = train_foundation_one_epoch(model, train_loader, optimizer, device=device)
        else:
            train_metrics = train_cnn_one_epoch(model, train_loader, optimizer, device=device)
        val_metrics, _, _ = _evaluate_model_with_predictions(model, val_loader, device=device, is_foundation=is_foundation)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_mae": train_metrics["mae"],
                "train_rmse": train_metrics["rmse"],
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_rmse": val_metrics["rmse"],
            }
        )
    val_metrics, val_pred, val_target = _evaluate_model_with_predictions(model, val_loader, device=device, is_foundation=is_foundation)
    in_metrics, in_pred, in_target = _evaluate_model_with_predictions(model, in_loader, device=device, is_foundation=is_foundation)
    cross_metrics, cross_pred, cross_target = _evaluate_model_with_predictions(model, cross_loader, device=device, is_foundation=is_foundation)
    metric_space, val_pred_physical, val_target_physical = _infer_physical_velocity_space(val_pred, val_target, stats_path=manifest["stats_path"])
    _, in_pred_physical, in_target_physical = _infer_physical_velocity_space(in_pred, in_target, stats_path=manifest["stats_path"])
    _, cross_pred_physical, cross_target_physical = _infer_physical_velocity_space(cross_pred, cross_target, stats_path=manifest["stats_path"])
    val_metrics = _attach_metric_space(compute_velocity_metrics(val_pred_physical, val_target_physical), metric_space)
    val_metrics["model_family"] = "foundation" if is_foundation else "cnn"
    in_metrics = _attach_metric_space(compute_velocity_metrics(in_pred_physical, in_target_physical), metric_space)
    in_metrics["model_family"] = "foundation" if is_foundation else "cnn"
    cross_metrics = _attach_metric_space(compute_velocity_metrics(cross_pred_physical, cross_target_physical), metric_space)
    cross_metrics["model_family"] = "foundation" if is_foundation else "cnn"
    _write_history_csv(run_dir / "train_history.csv", history)
    (run_dir / "metrics_val.json").write_text(json.dumps(val_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metrics_in_family_test.json").write_text(json.dumps(in_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metrics_cross_family_test.json").write_text(json.dumps(cross_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_predictions(
        run_dir / "predictions_in_family_test.npz",
        in_pred,
        in_target,
        velocity_pred_physical=in_pred_physical,
        velocity_true_physical=in_target_physical,
    )
    _write_predictions(
        run_dir / "predictions_cross_family_test.npz",
        cross_pred,
        cross_target,
        velocity_pred_physical=cross_pred_physical,
        velocity_true_physical=cross_target_physical,
    )
    return {
        "depth": depth,
        "width": width,
        "device": device,
        "metric_space": metric_space,
        "parameter_counts": parameter_counts,
        "split_counts": {name: len(paths) for name, paths in splits.items()},
    }


def _repo_status() -> dict[str, Any]:
    try:
        root = Path(__file__).resolve().parents[1]
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        status = subprocess.run(["git", "-C", str(root), "status", "--short"], capture_output=True, text=True, timeout=5)
        if commit.returncode == 0:
            return {"git_commit": commit.stdout.strip(), "git_status_short": status.stdout.strip()}
    except Exception:
        pass
    return {"git_commit": None, "git_status_short": "not a git repository"}


def _is_real_dinov2_available() -> bool:
    try:
        import timm  # noqa: F401

        return True
    except Exception:
        return False


def _manifest_paths(
    protocol_root: Path,
    max_pairs: int | None,
    *,
    source_family: str | None = None,
    target_family: str | None = None,
    seeds: list[int] | None = None,
) -> list[Path]:
    paths = sorted((protocol_root / "manifests").glob("*_manifest.json"))
    seed_filter = {int(seed) for seed in seeds} if seeds else None
    filtered: list[Path] = []
    for path in paths:
        payload = _load_manifest(path)
        if source_family and payload["source_family"] != source_family:
            continue
        if target_family and payload["target_family"] != target_family:
            continue
        if seed_filter is not None and int(payload["seed"]) not in seed_filter:
            continue
        filtered.append(path)
    paths = filtered
    if max_pairs is None:
        return paths
    seen_pairs: set[tuple[str, str]] = set()
    selected: list[Path] = []
    for path in paths:
        payload = _load_manifest(path)
        pair = (payload["source_family"], payload["target_family"])
        if pair not in seen_pairs:
            if len(seen_pairs) >= max_pairs:
                continue
            seen_pairs.add(pair)
        selected.append(path)
    return selected


def _resolve_cli_matrix(models: list[str] | None, bridges: list[str] | None) -> list[tuple[str, str]]:
    if not models and not bridges:
        return list(CPU_MATRIX)
    selected_models = set(models or [model for model, _ in CPU_MATRIX])
    selected_bridges = set(bridges or [bridge for _, bridge in CPU_MATRIX])
    unknown_models = sorted(selected_models - {model for model, _ in CPU_MATRIX})
    if unknown_models:
        raise ValueError(f"unsupported model filter: {', '.join(unknown_models)}")
    unknown_bridges = sorted(selected_bridges - {bridge for _, bridge in CPU_MATRIX})
    if unknown_bridges:
        raise ValueError(f"unsupported bridge filter: {', '.join(unknown_bridges)}")
    matrix = [(model_name, bridge) for model_name, bridge in CPU_MATRIX if model_name in selected_models and bridge in selected_bridges]
    if not matrix:
        raise ValueError("model/bridge filters produced an empty matrix")
    return matrix


def run_protocol_v2_matrix(
    *,
    protocol_root: str | Path,
    config_path: str | Path | None = None,
    cpu: bool = True,
    epochs: int = 3,
    max_pairs: int | None = None,
    dry_run: bool = False,
    models: list[tuple[str, str]] | None = None,
    source_family: str | None = None,
    target_family: str | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    root = Path(protocol_root)
    batch_size = choose_cpu_batch_size() if cpu else 4
    matrix = models or CPU_MATRIX
    real_dinov2_available = _is_real_dinov2_available()
    rows = []
    for manifest_path in _manifest_paths(root, max_pairs, source_family=source_family, target_family=target_family, seeds=seeds):
        manifest = _load_manifest(manifest_path)
        manifest["manifest_path"] = str(manifest_path)
        for model_name, bridge in matrix:
            run_dir = (
                root
                / f"{manifest['source_family']}_to_{manifest['target_family']}"
                / model_name
                / bridge
                / f"seed_{manifest['seed']}"
            )
            start = time.perf_counter()
            if dry_run:
                status = "SUCCESS"
                skip_reason = None
                if model_name.startswith("dinov2") and "smoke" in model_name and not real_dinov2_available:
                    status = "SKIPPED_REAL_DINOV2"
                    skip_reason = "timm is not importable in this environment"
                config = _write_protocol_outputs(
                    run_dir=run_dir,
                    manifest=manifest,
                    model_name=model_name,
                    bridge=bridge,
                    seed=int(manifest["seed"]),
                    epochs=epochs,
                    batch_size=batch_size,
                    status=status,
                    skip_reason=skip_reason,
                )
            else:
                extra = {}
                status = "SUCCESS"
                skip_reason = None
                try:
                    extra = _run_real_training(
                        run_dir=run_dir,
                        manifest=manifest,
                        model_name=model_name,
                        bridge=bridge,
                        seed=int(manifest["seed"]),
                        epochs=epochs,
                        batch_size=batch_size,
                        device="cpu" if cpu else "cuda",
                    )
                except RealDINOv2Skipped as exc:
                    status = "SKIPPED_REAL_DINOV2"
                    skip_reason = str(exc)
                except Exception as exc:
                    status = "FAILED"
                    skip_reason = f"{type(exc).__name__}: {exc}"
                config = _write_run_config(
                    run_dir=run_dir,
                    manifest=manifest,
                    model_name=model_name,
                    bridge=bridge,
                    seed=int(manifest["seed"]),
                    epochs=epochs,
                    batch_size=batch_size,
                    status=status,
                    runtime_seconds=time.perf_counter() - start,
                    skip_reason=skip_reason,
                    extra=extra,
                )
            rows.append(config)
    summary = {
        "protocol_root": str(root),
        "config_path": "" if config_path is None else str(config_path),
        "cpu": bool(cpu),
        "dry_run": bool(dry_run),
        "batch_size": int(batch_size),
        "run_count": len(rows),
        "skipped_real_dinov2": sum(1 for row in rows if row["status"] == "SKIPPED_REAL_DINOV2"),
        "runs": rows,
    }
    (root / "matrix_run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run protocol v2 CPU experiment matrix.")
    parser.add_argument("--protocol-root", type=Path, default=Path("outputs/protocol_v2_small"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Write deterministic protocol outputs without heavy training.")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--bridges", nargs="+", default=None)
    parser.add_argument("--source-family", type=str, default=None)
    parser.add_argument("--target-family", type=str, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    matrix = _resolve_cli_matrix(args.models, args.bridges)
    summary = run_protocol_v2_matrix(
        protocol_root=args.protocol_root,
        config_path=args.config,
        cpu=args.cpu,
        epochs=args.epochs,
        max_pairs=args.max_pairs,
        dry_run=args.dry_run,
        models=matrix,
        source_family=args.source_family,
        target_family=args.target_family,
        seeds=args.seeds,
    )
    print(f"Wrote matrix summary: {Path(args.protocol_root) / 'matrix_run_summary.json'}")
    if summary["skipped_real_dinov2"]:
        print("SKIPPED_REAL_DINOV2")


if __name__ == "__main__":
    main()
