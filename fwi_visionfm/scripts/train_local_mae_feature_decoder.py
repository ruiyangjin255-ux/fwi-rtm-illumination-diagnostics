from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from fwi_visionfm.evaluation.metrics import compute_velocity_metrics
from fwi_visionfm.models.decoders import build_decoder
from fwi_visionfm.scripts.run_protocol_v4_integrated_visual_search import _write_triplet_grid
from fwi_visionfm.training.losses import compute_loss_components
from fwi_visionfm.torch_backend import require_torch_backend


LOSS_PRESETS: dict[str, dict[str, float]] = {
    "default_l1": {"l1": 1.0},
    "gradient_l1": {"l1": 1.0, "gradient_l1": 0.2},
    "weak_gradient_l1": {"l1": 1.0, "gradient_l1": 0.02},
    "structure_loss": {"l1": 1.0, "gradient_l1": 0.2, "laplacian_l1": 0.1, "edge_weighted_l1": 0.2},
}


def _load_cache(path: Path) -> dict[str, Any]:
    torch = require_torch_backend()
    return torch.load(path, map_location="cpu")


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _iter_batches(payload: dict[str, Any], *, batch_size: int):
    features = _as_numpy(payload["features"]).astype(np.float32)
    target = _as_numpy(payload["target"]).astype(np.float32)
    records = _as_numpy(payload["records_preview"]).astype(np.float32)
    sample_ids = list(payload.get("sample_ids", [str(i) for i in range(int(features.shape[0]))]))
    for start in range(0, int(features.shape[0]), int(batch_size)):
        end = min(start + int(batch_size), int(features.shape[0]))
        yield features[start:end], target[start:end], records[start:end], sample_ids[start:end]


def _eval(decoder: Any, payload: dict[str, Any], *, device: str, loss_weights: dict[str, float]) -> tuple[dict[str, Any], np.ndarray]:
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
    prediction = np.concatenate(predictions, axis=0)
    target = _as_numpy(payload["target"]).astype(np.float32)
    metrics = compute_velocity_metrics(prediction, target)
    metrics["loss"] = float(np.mean(losses)) if losses else float(metrics["loss"])
    metrics["metric_space"] = "physical_velocity"
    return metrics, prediction


def _write_prediction(path: Path, *, prediction: np.ndarray, payload: dict[str, Any]) -> None:
    target = _as_numpy(payload["target"]).astype(np.float32)
    records = _as_numpy(payload["records_preview"]).astype(np.float32)
    sample_ids = list(payload.get("sample_ids", []))
    np.savez(
        path,
        prediction=prediction.astype(np.float32),
        target=target.astype(np.float32),
        velocity_pred_physical=prediction.astype(np.float32),
        velocity_true_physical=target.astype(np.float32),
        error_map_physical=(prediction - target).astype(np.float32),
        seismic_preview=records.astype(np.float32),
        sample_id=np.asarray(sample_ids),
        metric_space=np.asarray("physical_velocity"),
    )


def train_local_mae_decoder_from_cache(
    *,
    cache_root: str | Path,
    output_dir: str | Path,
    decoder_name: str,
    loss_name: str,
    epochs: int,
    batch_size: int,
    device: str,
) -> dict[str, Any]:
    torch = require_torch_backend()
    cache_dir = Path(cache_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    train_payload = _load_cache(cache_dir / "train_features.pt")
    val_payload = _load_cache(cache_dir / "val_features.pt")
    in_payload = _load_cache(cache_dir / "in_family_test_features.pt")
    cross_payload = _load_cache(cache_dir / "cross_family_test_features.pt")
    loss_weights = LOSS_PRESETS[loss_name]
    depth, width = map(int, metadata.get("target_shape", _as_numpy(train_payload["target"]).shape[-2:]))
    decoder = build_decoder(decoder_name, output_shape=(depth, width), base_channels=16, vmin=1500.0, vmax=4500.0).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1.0e-3)
    history = []
    for epoch in range(1, int(epochs) + 1):
        decoder.module.train()
        sums: dict[str, list[float]] = {}
        for features_np, target_np, _, _ in _iter_batches(train_payload, batch_size=batch_size):
            features = torch.as_tensor(features_np, dtype=torch.float32, device=device)
            target = torch.as_tensor(target_np, dtype=torch.float32, device=device)
            optimizer.zero_grad()
            pred = decoder(features)[:, 0]
            parts = compute_loss_components(pred, target, weights=loss_weights)
            parts["total_loss"].backward()
            optimizer.step()
            for key, value in parts.items():
                sums.setdefault(key, []).append(float(value.detach().cpu()))
        val_metrics, _ = _eval(decoder, val_payload, device=device, loss_weights=loss_weights)
        row = {"epoch": epoch, "val_loss": float(val_metrics["loss"]), "val_mae": float(val_metrics["mae"]), "val_rmse": float(val_metrics["rmse"])}
        for key, values in sorted(sums.items()):
            row[f"train_{key}"] = float(np.mean(values))
        history.append(row)
    val_metrics, val_pred = _eval(decoder, val_payload, device=device, loss_weights=loss_weights)
    in_metrics, in_pred = _eval(decoder, in_payload, device=device, loss_weights=loss_weights)
    cross_metrics, cross_pred = _eval(decoder, cross_payload, device=device, loss_weights=loss_weights)
    with (output / "train_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    (output / "metrics_val.json").write_text(json.dumps(val_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "metrics_in_family_test.json").write_text(json.dumps(in_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "metrics_cross_family_test.json").write_text(json.dumps(cross_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_prediction(output / "predictions_val.npz", prediction=val_pred, payload=val_payload)
    _write_prediction(output / "predictions_in_family_test.npz", prediction=in_pred, payload=in_payload)
    _write_prediction(output / "predictions_cross_family_test.npz", prediction=cross_pred, payload=cross_payload)
    _write_triplet_grid(output / "predictions_cross_family_test.npz", output / "prediction_grid.png", gradient=False)
    _write_triplet_grid(output / "predictions_cross_family_test.npz", output / "gradient_grid.png", gradient=True)
    config = {
        "model_name": "local_mae",
        "model_type": metadata.get("model_type", "pretrained_local_mae"),
        "bridge": metadata.get("bridge", ""),
        "mask_type": metadata.get("mask_type", "random_patch"),
        "decoder_name": decoder_name,
        "loss_name": loss_name,
        "seed": int(str(output).split("seed_")[-1]) if "seed_" in str(output) else 0,
        "epochs": int(epochs),
        "device": device,
        "metric_space": "physical_velocity",
        "status": "SUCCESS",
    }
    (output / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "run_log.txt").write_text("status=SUCCESS\n", encoding="utf-8")
    return {"status": "SUCCESS", "output_dir": str(output)}


def run_local_mae_decoder_matrix(
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
    losses: list[str] | None = None,
) -> dict[str, Any]:
    from fwi_visionfm.scripts.cache_local_mae_features import cache_local_mae_features_matrix
    from fwi_visionfm.scripts.cache_random_mae_encoder_features import cache_random_mae_features_matrix
    from fwi_visionfm.scripts.pretrain_local_seismic_mae import pretrain_local_mae_matrix

    root = Path(output_root)
    bridges = ["raw_envelope_spectrum3", "spectrogram_multiband"]
    mask_types = ["random_patch", "receiver_block", "trace_dropout", "frequency_band", "hybrid_seismic_mask"]
    pretrain_local_mae_matrix(
        data_root=data_root,
        output_root=root,
        source=source,
        bridges=bridges,
        train_size=train_size,
        val_size=val_size,
        seed=seed,
        epochs=5,
        batch_size=8,
        mask_ratio=0.75,
        device=device,
        mask_types=mask_types,
    )
    cache_local_mae_features_matrix(
        data_root=data_root,
        output_root=root,
        source=source,
        target=target,
        bridges=bridges,
        seed=seed,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        device=device,
        mask_types=mask_types,
    )
    cache_random_mae_features_matrix(
        data_root=data_root,
        output_root=root,
        source=source,
        target=target,
        bridges=["raw_envelope_spectrum3", "raw_spectrogram"],
        seed=seed,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        device=device,
        mask_types=["random_patch"],
    )
    rows = []
    selected_losses = list(losses or ["default_l1", "weak_gradient_l1"])
    successful_pretrained = []
    for bridge in bridges:
        for mask_type in mask_types:
            cache_root = root / "feature_cache" / "pretrained_local_mae" / bridge / mask_type / f"seed_{seed}"
            if not (cache_root / "metadata.json").exists():
                continue
            if "default_l1" in selected_losses:
                run_dir = root / "decoder_runs" / f"{source}_to_{target}" / "pretrained_local_mae" / bridge / mask_type / "unet_decoder" / "default_l1" / f"seed_{seed}"
                result = train_local_mae_decoder_from_cache(cache_root=cache_root, output_dir=run_dir, decoder_name="unet_decoder", loss_name="default_l1", epochs=epochs, batch_size=4, device=device)
                rows.append({"model_type": "pretrained_local_mae", "bridge": bridge, "mask_type": mask_type, "decoder_name": "unet_decoder", "loss_name": "default_l1", **result})
                metrics = json.loads((run_dir / "metrics_val.json").read_text(encoding="utf-8"))
                successful_pretrained.append({"bridge": bridge, "mask_type": mask_type, "score": -(float(metrics["mae"]) + float(metrics["rmse"]) + float(metrics["gradient_error"]) + float(metrics["edge_mae"])), "cache_root": cache_root})
    successful_pretrained.sort(key=lambda row: row["score"], reverse=True)
    for item in successful_pretrained[:2]:
        if "weak_gradient_l1" not in selected_losses:
            break
        run_dir = root / "decoder_runs" / f"{source}_to_{target}" / "pretrained_local_mae" / item["bridge"] / item["mask_type"] / "unet_decoder" / "weak_gradient_l1" / f"seed_{seed}"
        rows.append({"model_type": "pretrained_local_mae", "bridge": item["bridge"], "mask_type": item["mask_type"], "decoder_name": "unet_decoder", "loss_name": "weak_gradient_l1", **train_local_mae_decoder_from_cache(cache_root=item["cache_root"], output_dir=run_dir, decoder_name="unet_decoder", loss_name="weak_gradient_l1", epochs=epochs, batch_size=4, device=device)})
    for bridge in ["raw_envelope_spectrum3", "raw_spectrogram"]:
        cache_root = root / "random_encoder" / "feature_cache" / "random_mae_encoder" / bridge / "random_patch" / f"seed_{seed}"
        if not (cache_root / "metadata.json").exists():
            continue
        run_dir = root / "decoder_runs" / f"{source}_to_{target}" / "random_mae_encoder" / bridge / "random_patch" / "unet_decoder" / "default_l1" / f"seed_{seed}"
        rows.append({"model_type": "random_mae_encoder", "bridge": bridge, "mask_type": "random_patch", "decoder_name": "unet_decoder", "loss_name": "default_l1", **train_local_mae_decoder_from_cache(cache_root=cache_root, output_dir=run_dir, decoder_name="unet_decoder", loss_name="default_l1", epochs=epochs, batch_size=4, device=device)})
    summary = {"status": "SUCCESS", "rows": rows}
    (root / "local_mae_decoder_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train decoder-only velocity regression from local MAE features.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source", default="flatvel_a_subset2k")
    parser.add_argument("--target", default="curvevel_a_subset500")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-size", type=int, default=300)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--losses", nargs="+", default=["default_l1", "weak_gradient_l1"])
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_local_mae_decoder_matrix(
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
        losses=args.losses,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
