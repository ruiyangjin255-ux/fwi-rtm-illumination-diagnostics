from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.evaluation.metrics import compute_velocity_metrics
from fwi_visionfm.models.decoders import build_decoder
from fwi_visionfm.scripts.plot_structure_diagnostics import plot_structure_diagnostics
from fwi_visionfm.training.losses import compute_loss_components
from fwi_visionfm.torch_backend import require_torch_backend


LOSS_PRESETS: dict[str, dict[str, float]] = {
    "default_l1": {"l1": 1.0},
    "gradient_l1": {"l1": 1.0, "gradient_l1": 0.2},
    "structure_loss": {"l1": 1.0, "gradient_l1": 0.2, "laplacian_l1": 0.1, "edge_weighted_l1": 0.2},
}


def _load_cache(path: Path) -> dict[str, Any]:
    torch = require_torch_backend()
    payload = torch.load(path, map_location="cpu")
    return payload


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _iter_batches(payload: dict[str, Any], *, batch_size: int):
    features = _as_numpy(payload["features"]).astype(np.float32)
    target = _as_numpy(payload["target"]).astype(np.float32)
    records = _as_numpy(payload["records_preview"]).astype(np.float32)
    sample_ids = list(payload.get("sample_ids", [str(index) for index in range(int(features.shape[0]))]))
    for start in range(0, int(features.shape[0]), int(batch_size)):
        end = min(start + int(batch_size), int(features.shape[0]))
        yield features[start:end], target[start:end], records[start:end], sample_ids[start:end]


def _evaluate_decoder(decoder: Any, payload: dict[str, Any], *, device: str, loss_weights: dict[str, float]) -> tuple[dict[str, Any], np.ndarray]:
    torch = require_torch_backend()
    predictions = []
    losses = []
    with torch.no_grad():
        for features_np, target_np, _, _ in _iter_batches(payload, batch_size=16):
            features = torch.as_tensor(features_np, dtype=torch.float32, device=device)
            target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
            pred = decoder(features)[:, 0]
            parts = compute_loss_components(pred, target, weights=loss_weights)
            predictions.append(pred.detach().cpu().numpy().astype(np.float32))
            losses.append(float(parts["total_loss"].detach().cpu()))
    prediction = np.concatenate(predictions, axis=0) if predictions else np.zeros_like(_as_numpy(payload["target"]), dtype=np.float32)
    target = _as_numpy(payload["target"]).astype(np.float32)
    metrics = compute_velocity_metrics(prediction, target)
    metrics["loss"] = float(np.mean(losses)) if losses else float(metrics["loss"])
    metrics["metric_space"] = "physical_velocity"
    return metrics, prediction


def _write_prediction_npz(path: Path, *, prediction: np.ndarray, target: np.ndarray, records_preview: np.ndarray, sample_ids: list[str]) -> None:
    np.savez(
        path,
        prediction=prediction.astype(np.float32),
        target=target.astype(np.float32),
        velocity_pred_physical=prediction.astype(np.float32),
        velocity_true_physical=target.astype(np.float32),
        error_map_physical=(prediction - target).astype(np.float32),
        seismic_preview=records_preview.astype(np.float32),
        sample_id=np.asarray(sample_ids),
        metric_space=np.asarray("physical_velocity"),
    )


def train_ncs_feature_decoder(
    *,
    cache_root: str | Path,
    output_dir: str | Path,
    decoder_name: str,
    loss_name: str,
    loss_weights: dict[str, float] | None = None,
    epochs: int = 1,
    batch_size: int = 2,
    device: str = "cpu",
    allow_dummy_cache_smoke: bool = False,
) -> dict[str, Any]:
    torch = require_torch_backend()
    cache_dir = Path(cache_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8")) if (cache_dir / "metadata.json").exists() else {}
    train_payload = _load_cache(cache_dir / "train_features.pt")
    val_payload = _load_cache(cache_dir / "val_features.pt")
    in_payload = _load_cache(cache_dir / "in_family_test_features.pt")
    cross_payload = _load_cache(cache_dir / "cross_family_test_features.pt")
    is_real = bool(metadata.get("is_real_ncs_feature"))
    if not is_real and not allow_dummy_cache_smoke:
        config = {
            "decoder_name": decoder_name,
            "loss_name": loss_name,
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "device": device,
            "metric_space": "physical_velocity",
            "status": "DUMMY_CACHE_SMOKE",
            "skip_reason": "Dummy cache detected; set --allow-dummy-cache-smoke to run software-path smoke.",
        }
        (output / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        (output / "run_log.txt").write_text(config["skip_reason"] + "\n", encoding="utf-8")
        return {"status": "DUMMY_CACHE_SMOKE", "output_dir": str(output)}

    resolved_weights = dict(loss_weights or LOSS_PRESETS.get(loss_name, {"l1": 1.0}))
    target_shape = list(train_payload.get("target_shape") or metadata.get("target_shape") or _as_numpy(train_payload["target"]).shape[-2:])
    depth, width = map(int, target_shape[-2:])
    decoder = build_decoder(decoder_name, output_shape=(depth, width), base_channels=16, vmin=1500.0, vmax=4500.0).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1.0e-3)
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        decoder.module.train()
        sums: dict[str, list[float]] = {}
        for features_np, target_np, _, _ in _iter_batches(train_payload, batch_size=batch_size):
            features = torch.as_tensor(features_np, dtype=torch.float32, device=device)
            target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
            optimizer.zero_grad()
            pred = decoder(features)[:, 0]
            parts = compute_loss_components(pred, target, weights=resolved_weights)
            parts["total_loss"].backward()
            optimizer.step()
            for key, value in parts.items():
                sums.setdefault(key, []).append(float(value.detach().cpu()))
        val_metrics, _ = _evaluate_decoder(decoder, val_payload, device=device, loss_weights=resolved_weights)
        row = {"epoch": epoch, "val_loss": float(val_metrics["loss"]), "val_mae": float(val_metrics["mae"]), "val_rmse": float(val_metrics["rmse"])}
        for key, values in sorted(sums.items()):
            row[f"train_{key}"] = float(np.mean(values))
        history.append(row)

    val_metrics, val_pred = _evaluate_decoder(decoder, val_payload, device=device, loss_weights=resolved_weights)
    in_metrics, in_pred = _evaluate_decoder(decoder, in_payload, device=device, loss_weights=resolved_weights)
    cross_metrics, cross_pred = _evaluate_decoder(decoder, cross_payload, device=device, loss_weights=resolved_weights)

    with (output / "train_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    (output / "metrics_val.json").write_text(json.dumps(val_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "metrics_in_family_test.json").write_text(json.dumps(in_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "metrics_cross_family_test.json").write_text(json.dumps(cross_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    _write_prediction_npz(
        output / "predictions_val.npz",
        prediction=val_pred,
        target=_as_numpy(val_payload["target"]).astype(np.float32),
        records_preview=_as_numpy(val_payload["records_preview"]).astype(np.float32),
        sample_ids=list(val_payload.get("sample_ids", [])),
    )
    _write_prediction_npz(
        output / "predictions_in_family_test.npz",
        prediction=in_pred,
        target=_as_numpy(in_payload["target"]).astype(np.float32),
        records_preview=_as_numpy(in_payload["records_preview"]).astype(np.float32),
        sample_ids=list(in_payload.get("sample_ids", [])),
    )
    _write_prediction_npz(
        output / "predictions_cross_family_test.npz",
        prediction=cross_pred,
        target=_as_numpy(cross_payload["target"]).astype(np.float32),
        records_preview=_as_numpy(cross_payload["records_preview"]).astype(np.float32),
        sample_ids=list(cross_payload.get("sample_ids", [])),
    )
    plot_structure_diagnostics(
        predictions_path=output / "predictions_cross_family_test.npz",
        metrics_path=output / "metrics_cross_family_test.json",
        output_dir=output / "structure_diagnostics",
        prefix=f"ncs_feature_decoder_{decoder_name}_{loss_name}_cross_family_test",
    )
    status = "SUCCESS" if is_real else "DUMMY_CACHE_SMOKE"
    config = {
        "decoder_name": decoder_name,
        "loss_name": loss_name,
        "loss_weights": resolved_weights,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "device": device,
        "metric_space": "physical_velocity",
        "status": status,
        "is_real_ncs_feature": is_real,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "run_log.txt").write_text(f"status={status}\n", encoding="utf-8")
    return {"status": status, "output_dir": str(output), "config": config}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train decoder from cached NCS features.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decoder", default="unet_decoder")
    parser.add_argument("--loss", dest="loss_name", default="default_l1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-dummy-cache-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_ncs_feature_decoder(
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        decoder_name=args.decoder,
        loss_name=args.loss_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        allow_dummy_cache_smoke=args.allow_dummy_cache_smoke,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
