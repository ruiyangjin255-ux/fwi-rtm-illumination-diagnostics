from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.evaluation.metrics import compute_velocity_metrics
from fwi_visionfm.models.parameter_utils import count_parameters
from fwi_visionfm.peft import LoRAConfig
from fwi_visionfm.scripts.build_protocol_v2_splits import build_protocol_v2_splits
from fwi_visionfm.scripts.plot_structure_diagnostics import plot_structure_diagnostics
from fwi_visionfm.torch_backend import require_torch_backend
from fwi_visionfm.torch_backend.data import build_torch_dataloader
from fwi_visionfm.torch_backend.model import FrozenFoundationFWI, FwiVisionFmTorchBaseline
from fwi_visionfm.torch_backend.train import count_model_parameters, set_torch_seed
from fwi_visionfm.training.losses import compute_loss_components

LOSS_PRESETS: dict[str, dict[str, float]] = {
    "default_l1": {"l1": 1.0},
    "gradient_l1": {"l1": 1.0, "gradient_l1": 0.2},
    "structure_loss": {"l1": 1.0, "gradient_l1": 0.2, "laplacian_l1": 0.1, "edge_weighted_l1": 0.2},
}


class RealDINOv2Skipped(RuntimeError):
    pass


def _safe_count_model_parameters(model: Any) -> dict[str, Any]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        try:
            count = int(parameter.numel())
        except ValueError:
            continue
        total += count
        if bool(getattr(parameter, "requires_grad", False)):
            trainable += count
    ratio = float(trainable / total) if total > 0 else 0.0
    return {"total_parameters": total, "trainable_parameters": trainable, "trainable_ratio": ratio}


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_paths(rows: list[dict[str, Any]]) -> list[Path]:
    return [Path(row.get("path") or row["data_file"]) for row in rows]


def _split_paths(manifest: dict[str, Any]) -> dict[str, list[Path]]:
    return {
        "train": _sample_paths(manifest["train_samples"]),
        "val": _sample_paths(manifest["val_samples"]),
        "in_family_test": _sample_paths(manifest["in_family_test_samples"]),
        "cross_family_test": _sample_paths(manifest["cross_family_test_samples"]),
    }


def _first_velocity_shape(paths: list[Path]) -> tuple[int, int]:
    with np.load(paths[0]) as payload:
        velocity = np.asarray(payload["velocity"], dtype=np.float32)
    return int(velocity.shape[-2]), int(velocity.shape[-1])


def _read_velocity_bounds(stats_path: str | Path) -> tuple[float, float]:
    payload = json.loads(Path(stats_path).read_text(encoding="utf-8"))
    velocity = payload.get("velocity", {})
    if "min" in velocity and "max" in velocity:
        return float(velocity["min"]), float(velocity["max"])
    if "target_min" in payload and "target_max" in payload:
        return float(payload["target_min"]), float(payload["target_max"])
    return 1500.0, 4500.0


def _evaluate_model_with_predictions(model: Any, dataloader: Any, *, device: str, loss_weights: dict[str, float]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    torch = require_torch_backend()
    model.eval()
    predictions = []
    targets = []
    records_preview = []
    total_losses = []
    with torch.no_grad():
        for batch in dataloader:
            records = batch["records"].to(device)
            target = batch["velocity"].to(device)
            source_positions = batch["source_positions"].to(device)
            pred = model(records, source_positions)
            parts = compute_loss_components(pred, target, weights=loss_weights)
            total_losses.append(float(parts["total_loss"].detach().cpu()))
            predictions.append(pred.detach().cpu().numpy().astype(np.float32))
            targets.append(target.detach().cpu().numpy().astype(np.float32))
            records_preview.append(batch["records"].cpu().numpy().astype(np.float32))
    prediction = np.concatenate(predictions, axis=0)
    target = np.concatenate(targets, axis=0)
    preview = np.concatenate(records_preview, axis=0)
    metrics = compute_velocity_metrics(prediction, target)
    metrics["loss"] = float(np.mean(total_losses)) if total_losses else float(metrics["loss"])
    metrics["metric_space"] = "physical_velocity"
    return metrics, prediction, target, preview


def _write_predictions(path: Path, prediction: np.ndarray, target: np.ndarray, records_preview: np.ndarray) -> None:
    np.savez(
        path,
        prediction=prediction.astype(np.float32),
        target=target.astype(np.float32),
        velocity_pred=prediction.astype(np.float32),
        velocity_true=target.astype(np.float32),
        error_map=(prediction - target).astype(np.float32),
        velocity_pred_physical=prediction.astype(np.float32),
        velocity_true_physical=target.astype(np.float32),
        error_map_physical=(prediction - target).astype(np.float32),
        seismic_preview=records_preview.astype(np.float32),
    )


def _write_history(path: Path, history: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def _build_model(
    *,
    model_name: str,
    bridge: str,
    decoder_name: str,
    depth: int,
    width: int,
    vmin: float,
    vmax: float,
    device: str,
) -> tuple[Any, bool, dict[str, Any] | None, bool]:
    if model_name == "cnn_baseline":
        model = FwiVisionFmTorchBaseline(
            channels=("raw",),
            depth=depth,
            width=width,
            bridge_mode="simple",
            aggregation="mean",
            feature_dim=64,
            vmin=vmin,
            vmax=vmax,
            decoder_name=decoder_name,
            bridge_name=model_name == "cnn_baseline" and bridge or None,
            bridge_config={"output_size": [64, 64]},
        ).to(device)
        return model, False, None, False
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
            bridge_feature_mode=bridge,
            depth=depth,
            width=width,
            vmin=vmin,
            vmax=vmax,
            device=device,
            print_parameter_report=False,
            decoder_name=decoder_name,
        ).to(device)
        total, trainable, ratio = count_parameters(model.module)
        return model, True, {"total_parameters": total, "trainable_parameters": trainable, "trainable_ratio": ratio}, False
    if model_name == "dinov2_lora_smoke":
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
            bridge_feature_mode=bridge,
            depth=depth,
            width=width,
            vmin=vmin,
            vmax=vmax,
            device=device,
            print_parameter_report=False,
            decoder_name=decoder_name,
        ).to(device)
        total, trainable, ratio = count_parameters(model.module)
        return model, True, {"total_parameters": total, "trainable_parameters": trainable, "trainable_ratio": ratio}, True
    raise ValueError(f"unsupported model: {model_name}")


def _run_single(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    model_name: str,
    bridge: str,
    decoder_name: str,
    loss_name: str,
    loss_weights: dict[str, float],
    epochs: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    torch = require_torch_backend()
    set_torch_seed(int(manifest["seed"]))
    splits = _split_paths(manifest)
    depth, width = _first_velocity_shape(splits["train"])
    vmin, vmax = _read_velocity_bounds(manifest["stats_path"])
    run_dir.mkdir(parents=True, exist_ok=True)
    model, is_foundation, parameter_counts, is_probe = _build_model(
        model_name=model_name,
        bridge=bridge,
        decoder_name=decoder_name,
        depth=depth,
        width=width,
        vmin=vmin,
        vmax=vmax,
        device=device,
    )
    train_loader = build_torch_dataloader(splits["train"], batch_size=batch_size, shuffle=True, seed=int(manifest["seed"]))
    val_loader = build_torch_dataloader(splits["val"], batch_size=batch_size, shuffle=False, seed=int(manifest["seed"]))
    in_loader = build_torch_dataloader(splits["in_family_test"], batch_size=batch_size, shuffle=False, seed=int(manifest["seed"]))
    cross_loader = build_torch_dataloader(splits["cross_family_test"], batch_size=batch_size, shuffle=False, seed=int(manifest["seed"]))
    first_batch = next(iter(train_loader))
    with torch.no_grad():
        _ = model(first_batch["records"].to(device), first_batch["source_positions"].to(device))
    if parameter_counts is None:
        parameter_counts = _safe_count_model_parameters(model)
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1.0e-3)
    history: list[dict[str, Any]] = []
    train_epochs = 1 if model_name.startswith("dinov2") else int(epochs)
    for epoch in range(1, train_epochs + 1):
        model.train()
        component_sums: dict[str, list[float]] = {}
        for batch in train_loader:
            records = batch["records"].to(device)
            target = batch["velocity"].to(device)
            source_positions = batch["source_positions"].to(device)
            optimizer.zero_grad()
            pred = model(records, source_positions)
            parts = compute_loss_components(pred, target, weights=loss_weights)
            parts["total_loss"].backward()
            optimizer.step()
            for key, value in parts.items():
                component_sums.setdefault(key, []).append(float(value.detach().cpu()))
        val_metrics, _, _, _ = _evaluate_model_with_predictions(model, val_loader, device=device, loss_weights=loss_weights)
        row = {"epoch": epoch}
        for key, values in sorted(component_sums.items()):
            row[f"train_{key}"] = float(np.mean(values))
        row["val_loss"] = float(val_metrics["loss"])
        row["val_mae"] = float(val_metrics["mae"])
        row["val_rmse"] = float(val_metrics["rmse"])
        row["val_gradient_error"] = float(val_metrics["gradient_error"])
        row["val_edge_mae"] = float(val_metrics["edge_mae"])
        history.append(row)
    val_metrics, _, _, _ = _evaluate_model_with_predictions(model, val_loader, device=device, loss_weights=loss_weights)
    in_metrics, in_pred, in_target, in_preview = _evaluate_model_with_predictions(model, in_loader, device=device, loss_weights=loss_weights)
    cross_metrics, cross_pred, cross_target, cross_preview = _evaluate_model_with_predictions(model, cross_loader, device=device, loss_weights=loss_weights)
    _write_history(run_dir / "train_history.csv", history)
    (run_dir / "metrics_val.json").write_text(json.dumps(val_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metrics_in_family_test.json").write_text(json.dumps(in_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metrics_cross_family_test.json").write_text(json.dumps(cross_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_predictions(run_dir / "predictions_in_family_test.npz", in_pred, in_target, in_preview)
    _write_predictions(run_dir / "predictions_cross_family_test.npz", cross_pred, cross_target, cross_preview)
    prefix_base = f"{manifest['source_family']}_{manifest['target_family']}_{model_name}_{bridge}_{decoder_name}_{loss_name}_seed{manifest['seed']}"
    plot_structure_diagnostics(
        predictions_path=run_dir / "predictions_in_family_test.npz",
        metrics_path=run_dir / "metrics_in_family_test.json",
        output_dir=run_dir / "structure_diagnostics",
        prefix=f"{prefix_base}_in_family_test",
    )
    plot_structure_diagnostics(
        predictions_path=run_dir / "predictions_cross_family_test.npz",
        metrics_path=run_dir / "metrics_cross_family_test.json",
        output_dir=run_dir / "structure_diagnostics",
        prefix=f"{prefix_base}_cross_family_test",
    )
    return {
        "status": "SUCCESS",
        "metric_space": "physical_velocity",
        "parameter_counts": parameter_counts,
        "is_probe": is_probe,
        "actual_epochs": train_epochs,
        "bridge_metadata": getattr(getattr(model, "registry_bridge", None), "last_metadata", {}),
    }


def _manifest_for(root: Path, *, source: str, target: str, seed: int) -> Path:
    matches = list((root / "manifests").glob(f"{source}_to_{target}_seed{seed}_manifest.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one manifest for {source}->{target} seed {seed}, got {len(matches)}")
    return matches[0]


def _matrix_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for model_name in ("cnn_baseline", "vit_tiny_scratch"):
        for bridge in ("raw_repeat3", "raw_spectrogram"):
            for decoder_name in ("simple_bounded_decoder", "unet_decoder"):
                entries.append(
                    {"model_name": model_name, "bridge": bridge, "decoder_name": decoder_name, "loss_name": "default_l1", "epochs": 3}
                )
    for loss_name in ("gradient_l1", "structure_loss"):
        entries.append(
            {
                "model_name": "vit_tiny_scratch",
                "bridge": "raw_spectrogram",
                "decoder_name": "unet_decoder",
                "loss_name": loss_name,
                "epochs": 3,
            }
        )
    for decoder_name in ("simple_bounded_decoder", "unet_decoder"):
        entries.append(
            {
                "model_name": "dinov2_lora_smoke",
                "bridge": "raw_spectrogram",
                "decoder_name": decoder_name,
                "loss_name": "default_l1",
                "epochs": 1,
            }
        )
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for entry in entries:
        key = (entry["model_name"], entry["bridge"], entry["decoder_name"], entry["loss_name"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def run_protocol_v3_structure_matrix(
    *,
    data_root: str | Path,
    output_root: str | Path,
    source: str,
    target: str,
    seed: int,
    train_size: int,
    val_size: int,
    test_size: int,
    epochs: int,
    device: str,
) -> dict[str, Any]:
    root = Path(output_root)
    build_protocol_v2_splits(
        data_root=data_root,
        output_root=root,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seeds=[seed],
    )
    manifest_path = _manifest_for(root, source=source, target=target, seed=seed)
    manifest = _load_manifest(manifest_path)
    rows = []
    for entry in _matrix_entries():
        run_dir = root / f"{source}_to_{target}" / entry["model_name"] / entry["bridge"] / entry["decoder_name"] / entry["loss_name"] / f"seed_{seed}"
        start = time.perf_counter()
        status = "SUCCESS"
        skip_reason = ""
        extra: dict[str, Any] = {}
        try:
            extra = _run_single(
                run_dir=run_dir,
                manifest=manifest,
                model_name=entry["model_name"],
                bridge=entry["bridge"],
                decoder_name=entry["decoder_name"],
                loss_name=entry["loss_name"],
                loss_weights=LOSS_PRESETS[entry["loss_name"]],
                epochs=entry["epochs"] if entry["model_name"].startswith("dinov2") else epochs,
                batch_size=4,
                device=device,
            )
        except RealDINOv2Skipped as exc:
            status = "SKIPPED_REAL_DINOV2"
            skip_reason = str(exc)
        except Exception as exc:
            status = "FAILED"
            skip_reason = f"{type(exc).__name__}: {exc}"
        config = {
            "protocol": "protocol_v3_structure_aware_decoder_loss",
            "source_family": source,
            "target_family": target,
            "model_name": entry["model_name"],
            "bridge": entry["bridge"],
            "decoder_name": entry["decoder_name"],
            "loss_name": entry["loss_name"],
            "loss_weights": LOSS_PRESETS[entry["loss_name"]],
            "seed": int(seed),
            "epochs": int(entry["epochs"] if entry["model_name"].startswith("dinov2") else epochs),
            "batch_size": 4,
            "device": device,
            "metric_space": extra.get("metric_space", "physical_velocity"),
            "status": status,
            "skip_reason": skip_reason,
            "runtime_seconds": time.perf_counter() - start,
            "is_probe": bool(extra.get("is_probe", False)),
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        rows.append(config)
    summary = {"run_count": len(rows), "runs": rows, "output_root": str(root)}
    (root / "matrix_run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Protocol V3 structure-aware decoder/loss CPU matrix.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=300)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_protocol_v3_structure_matrix(
        data_root=args.data_root,
        output_root=args.output_root,
        source=args.source,
        target=args.target,
        seed=args.seed,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        epochs=args.epochs,
        device=args.device,
    )
    print(f"Wrote matrix summary: {Path(args.output_root) / 'matrix_run_summary.json'}")
    print(f"run_count={summary['run_count']}")


if __name__ == "__main__":
    main()
