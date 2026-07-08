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


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _iter_batches(features: np.ndarray, target: np.ndarray, records_preview: np.ndarray, *, batch_size: int):
    for start in range(0, int(features.shape[0]), int(batch_size)):
        end = min(start + int(batch_size), int(features.shape[0]))
        yield features[start:end], target[start:end], records_preview[start:end]


def _evaluate_decoder(decoder: Any, payload: dict[str, np.ndarray], *, device: str, loss_weights: dict[str, float]) -> tuple[dict[str, Any], np.ndarray]:
    torch = require_torch_backend()
    predictions = []
    losses = []
    with torch.no_grad():
        for features_np, target_np, _ in _iter_batches(payload["features"], payload["target"], payload["records_preview"], batch_size=16):
            features = torch.as_tensor(features_np, dtype=torch.float32, device=device)
            target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
            pred = decoder(features)[:, 0]
            loss_parts = compute_loss_components(pred, target, weights=loss_weights)
            losses.append(float(loss_parts["total_loss"].detach().cpu()))
            predictions.append(pred.detach().cpu().numpy().astype(np.float32))
    prediction = np.concatenate(predictions, axis=0)
    target = np.asarray(payload["target"], dtype=np.float32)
    metrics = compute_velocity_metrics(prediction, target)
    metrics["loss"] = float(np.mean(losses)) if losses else float(metrics["loss"])
    metrics["metric_space"] = "physical_velocity"
    return metrics, prediction


def _write_prediction_npz(path: Path, *, prediction: np.ndarray, payload: dict[str, np.ndarray]) -> None:
    sample_ids = payload.get("sample_ids", np.asarray([str(i) for i in range(int(prediction.shape[0]))]))
    np.savez(
        path,
        prediction=prediction.astype(np.float32),
        target=payload["target"].astype(np.float32),
        velocity_pred_physical=prediction.astype(np.float32),
        velocity_true_physical=payload["target"].astype(np.float32),
        error_map_physical=(prediction - payload["target"]).astype(np.float32),
        seismic_preview=payload["records_preview"].astype(np.float32),
        sample_id=sample_ids,
        metric_space=np.asarray("physical_velocity"),
    )


def train_feature_decoder(
    *,
    cache_root: str | Path,
    output_dir: str | Path,
    decoder_name: str,
    loss_name: str,
    loss_weights: dict[str, float] | None = None,
    epochs: int = 1,
    batch_size: int = 2,
    device: str = "cpu",
) -> dict[str, Any]:
    torch = require_torch_backend()
    cache_dir = Path(cache_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_payload = _load_cache(cache_dir / "train_features.npz")
    val_payload = _load_cache(cache_dir / "val_features.npz")
    in_payload = _load_cache(cache_dir / "in_family_test_features.npz")
    cross_payload = _load_cache(cache_dir / "cross_family_test_features.npz")
    resolved_weights = dict(loss_weights or LOSS_PRESETS.get(loss_name, {"l1": 1.0}))
    depth, width = map(int, train_payload["target"].shape[-2:])
    decoder = build_decoder(decoder_name, output_shape=(depth, width), base_channels=16, vmin=1500.0, vmax=4500.0).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1.0e-3)
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        decoder.module.train()
        component_sums: dict[str, list[float]] = {}
        for features_np, target_np, _ in _iter_batches(train_payload["features"], train_payload["target"], train_payload["records_preview"], batch_size=batch_size):
            features = torch.as_tensor(features_np, dtype=torch.float32, device=device)
            target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
            optimizer.zero_grad()
            pred = decoder(features)[:, 0]
            loss_parts = compute_loss_components(pred, target, weights=resolved_weights)
            loss_parts["total_loss"].backward()
            optimizer.step()
            for name, value in loss_parts.items():
                component_sums.setdefault(name, []).append(float(value.detach().cpu()))
        val_metrics, _ = _evaluate_decoder(decoder, val_payload, device=device, loss_weights=resolved_weights)
        row = {"epoch": epoch}
        for name, values in sorted(component_sums.items()):
            row[f"train_{name}"] = float(np.mean(values))
        row["val_loss"] = float(val_metrics["loss"])
        row["val_mae"] = float(val_metrics["mae"])
        row["val_rmse"] = float(val_metrics["rmse"])
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
    _write_prediction_npz(output / "predictions_val.npz", prediction=val_pred, payload=val_payload)
    _write_prediction_npz(output / "predictions_in_family_test.npz", prediction=in_pred, payload=in_payload)
    _write_prediction_npz(output / "predictions_cross_family_test.npz", prediction=cross_pred, payload=cross_payload)
    plot_structure_diagnostics(
        predictions_path=output / "predictions_cross_family_test.npz",
        metrics_path=output / "metrics_cross_family_test.json",
        output_dir=output / "structure_diagnostics",
        prefix=f"feature_decoder_{decoder_name}_{loss_name}_cross_family_test",
    )
    config = {
        "decoder_name": decoder_name,
        "loss_name": loss_name,
        "loss_weights": resolved_weights,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "device": device,
        "metric_space": "physical_velocity",
        "status": "SUCCESS",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"status": "SUCCESS", "output_dir": str(output), "config": config}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train decoder only from cached foundation features.")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--decoder", default="unet_decoder")
    parser.add_argument("--loss", dest="loss_name", default="default_l1")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = train_feature_decoder(
        cache_root=args.cache_root,
        output_dir=args.output_dir,
        decoder_name=args.decoder,
        loss_name=args.loss_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
