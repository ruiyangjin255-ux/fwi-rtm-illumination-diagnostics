# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import yaml

from fwi_visionfm.models.geometry_aware_trace_bridge import GeometryAwareTraceBridge
from fwi_visionfm.models.protocol_v11_common_decoder import build_protocol_v11_decoder
from fwi_visionfm.models.seismic_backbones.ncs_backbone import load_ncs_model
from fwi_visionfm.models.seismic_bridge import SeismicToVisionBridge
from fwi_visionfm.models.vision_backbones import build_vision_backbone
from fwi_visionfm.datasets import load_npz_sample
from fwi_visionfm.torch_backend import require_torch_backend


def _parameter_count(parameters: list[Any]) -> int:
    from torch.nn.parameter import UninitializedParameter

    return int(sum(parameter.numel() for parameter in parameters if not isinstance(parameter, UninitializedParameter)))


def build_geometry_bridge_optimizer_report(*, bridge: GeometryAwareTraceBridge, decoder: Any, learning_rate: float) -> dict[str, Any]:
    torch = require_torch_backend()
    bridge_parameters = list(dict.fromkeys(parameter for parameter in bridge.condition_encoder.module.parameters()))
    bridge_parameters.extend(list(dict.fromkeys(parameter for parameter in bridge.fusion.layer_norm.parameters())))
    bridge_parameters.extend(list(dict.fromkeys(parameter for parameter in bridge.fusion.gate.parameters())))
    bridge_parameters.extend(list(dict.fromkeys(parameter for parameter in bridge.fusion.gamma.parameters())))
    bridge_parameters.extend(list(dict.fromkeys(parameter for parameter in bridge.fusion.beta.parameters())))
    bridge_parameters.extend(list(dict.fromkeys(parameter for parameter in bridge.trace_encoder.module.parameters())))
    bridge_parameters.extend(list(dict.fromkeys(parameter for parameter in bridge.shot_encoder.module.parameters())))
    decoder_parameters = list(dict.fromkeys(parameter for parameter in decoder.parameters()))
    trainable = list(dict.fromkeys([parameter for parameter in bridge_parameters + decoder_parameters if parameter.requires_grad]))
    optimizer = torch.optim.Adam(trainable, lr=float(learning_rate))
    optimizer_parameters = list(dict.fromkeys(parameter for group in optimizer.param_groups for parameter in group["params"]))
    optimizer_ids = {id(parameter) for parameter in optimizer_parameters}
    decoder_ids = {id(parameter) for parameter in decoder_parameters if parameter.requires_grad}
    film_parameters = (
        list(bridge.fusion.gate.parameters())
        + list(bridge.fusion.gamma.parameters())
        + list(bridge.fusion.beta.parameters())
    )
    return {
        "optimizer": optimizer,
        "trainable_parameters": _parameter_count(trainable),
        "optimizer_parameters": _parameter_count(optimizer_parameters),
        "film_parameters": _parameter_count(film_parameters),
        "decoder_fully_registered": decoder_ids.issubset(optimizer_ids),
    }


def write_protocol_v14_prediction_npz(path: str | Path, *, prediction: np.ndarray, target: np.ndarray, sample_ids: list[str], metadata: dict[str, Any]) -> Path:
    pred = np.asarray(prediction, dtype=np.float32)
    truth = np.asarray(target, dtype=np.float32)
    if pred.ndim == 3:
        pred = pred[:, None]
    if truth.ndim == 3:
        truth = truth[:, None]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        velocity_pred_physical=pred,
        velocity_true_physical=truth,
        error_map_physical=(pred - truth).astype(np.float32),
        prediction=pred,
        target=truth,
        sample_id=np.asarray(sample_ids, dtype=str),
        model_id=np.asarray(str(metadata["model_id"])),
        bridge_name=np.asarray(str(metadata["bridge_name"])),
        geometry_mode=np.asarray(str(metadata["geometry_mode"])),
        geometry_provenance=np.asarray(str(metadata["geometry_provenance"])),
        trace_context_radius=np.asarray(int(metadata["trace_context_radius"])),
        use_shot_global_context=np.asarray(bool(metadata["use_shot_global_context"])),
        use_multiscale_context=np.asarray(bool(metadata["use_multiscale_context"])),
        source_family=np.asarray(str(metadata["source_family"])),
        target_family=np.asarray(str(metadata["target_family"])),
        seed=np.asarray(int(metadata["seed"])),
        metric_space=np.asarray(str(metadata["metric_space"])),
        is_real_feature=np.asarray(bool(metadata["is_real_feature"])),
    )
    return output


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _bool(value: Any) -> bool:
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _read_matrix(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _manifest_map(root: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for path in sorted(root.glob("*_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if {"source_family", "target_family", "seed"}.issubset(payload):
            result[(payload["source_family"], payload["target_family"], int(payload["seed"]))] = payload
    return result


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _copy_reused_run(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.is_file():
            _link_or_copy(path, target / path.name)


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def _load_matplotlib():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _gradient_mag(image: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(np.asarray(image, dtype=np.float32))
    return np.sqrt(gx * gx + gy * gy)


def _load_grid_arrays(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(npz_path, allow_pickle=True) as payload:
        pred = np.asarray(payload["velocity_pred_physical"] if "velocity_pred_physical" in payload else payload["prediction"], dtype=np.float32)
        target = np.asarray(payload["velocity_true_physical"] if "velocity_true_physical" in payload else payload["target"], dtype=np.float32)
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0]
    return pred, target


def _write_grid_png(path: Path, *, npz_path: Path, gradient: bool) -> None:
    pred, target = _load_grid_arrays(npz_path)
    sample_mae = np.mean(np.abs(pred - target), axis=(1, 2))
    order = np.argsort(sample_mae)
    picks = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    labels = ["best", "median", "worst"]
    plt = _load_matplotlib()
    fig, axes = plt.subplots(len(picks), 3, figsize=(11.5, 3.4 * len(picks)), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row_idx, (label, index) in enumerate(zip(labels, picks)):
        pred_map = _gradient_mag(pred[index]) if gradient else pred[index]
        target_map = _gradient_mag(target[index]) if gradient else target[index]
        error_map = np.abs(pred_map - target_map).astype(np.float32)
        value_min = 0.0 if gradient else float(min(np.min(pred_map), np.min(target_map)))
        value_max = float(max(np.max(pred_map), np.max(target_map)))
        error_max = float(np.max(error_map)) if float(np.max(error_map)) > 0.0 else 1.0
        panels = [
            (target_map, f"{label} target", "viridis", value_min, value_max),
            (pred_map, f"{label} prediction", "viridis", value_min, value_max),
            (error_map, f"{label} abs error", "magma", 0.0, error_max),
        ]
        for col_idx, (array, title, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(array, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("gradient diagnostics" if gradient else "velocity prediction diagnostics", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _trace_cache_dir(*, root: Path, transfer_id: str, method_key: str, seed: int) -> Path:
    return root / transfer_id / method_key / f"seed_{seed}"


def _is_complete_v14_success(run_dir: Path) -> bool:
    required = {
        "config.json",
        "config_hash.txt",
        "model_card.json",
        "geometry_metadata.json",
        "train_history.csv",
        "metrics_val.json",
        "metrics_in_family_test.json",
        "metrics_cross_family_test.json",
        "predictions_in_family_test.npz",
        "predictions_cross_family_test.npz",
        "prediction_grid.png",
        "gradient_grid.png",
        "run_log.txt",
    }
    if not all((run_dir / name).is_file() for name in required):
        return False
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return config.get("status") == "SUCCESS"


def _velocity_bounds(manifest: dict[str, Any]) -> tuple[float, float]:
    stats = json.loads(Path(manifest["stats_path"]).read_text(encoding="utf-8"))
    velocity = stats.get("velocity", {})
    return float(velocity.get("min", 1500.0)), float(velocity.get("max", 4500.0))


def _trace_window_records(records: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return records[:, :, :, None, :]
    padded = np.pad(records, ((0, 0), (0, 0), (radius, radius), (0, 0)), mode="edge")
    windows = []
    for index in range(records.shape[2]):
        windows.append(padded[:, :, index : index + 2 * radius + 1, :][:, :, :, None, :])
    return np.concatenate(windows, axis=3)


def _compute_multiscale_feature(records: np.ndarray) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(records, axis=-1))
    bands = np.array_split(spectrum, 3, axis=-1)
    pooled = [band.mean(axis=-1, keepdims=True) for band in bands]
    while len(pooled) < 3:
        pooled.append(np.zeros_like(pooled[0]))
    return np.concatenate(pooled[:3], axis=-1).astype(np.float32)


def _canonical_geometry(records: np.ndarray, source_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    batch, shots, receivers, _ = records.shape
    xs = np.repeat(source_positions[:, :, None], receivers, axis=2)
    xr = np.repeat(np.linspace(0.0, 1.0, receivers, dtype=np.float32)[None, None, :], batch * shots, axis=0).reshape(batch, shots, receivers)
    signed = xr - xs
    abs_offset = np.abs(signed)
    midpoint = 0.5 * (xr + xs)
    shot_index = np.repeat((np.arange(shots, dtype=np.float32) / max(shots - 1, 1))[None, :, None], batch * receivers, axis=0).reshape(batch, shots, receivers)
    receiver_index = np.repeat((np.arange(receivers, dtype=np.float32) / max(receivers - 1, 1))[None, None, :], batch * shots, axis=0).reshape(batch, shots, receivers)
    g_trace = np.stack([xs, xr, signed, abs_offset, midpoint, shot_index, receiver_index], axis=-1).astype(np.float32)
    g_time = np.zeros((batch, shots, receivers, 1, 3), dtype=np.float32)
    g_time[..., 0, 0] = 0.5
    return g_trace, g_time


class GeometryAwareTraceFeatureRegressor:
    def __init__(self, *, token_dim: int, output_shape: tuple[int, int], bridge_row: dict[str, Any], vmin: float, vmax: float, base_channels: int, geometry_provenance: str) -> None:
        self.bridge = GeometryAwareTraceBridge(
            token_dim=token_dim,
            trace_context_radius=int(bridge_row["trace_context_radius"]),
            geometry_condition_dim=10,
            trace_context_dim=64,
            shot_context_dim=64,
            use_trace_context=bridge_row["bridge_id"] in {"B2", "B3"},
            use_shot_global_context=_bool(bridge_row["use_shot_global_context"]),
            use_multiscale_context=_bool(bridge_row["use_multiscale_context"]),
            geometry_provenance=geometry_provenance,
        )
        self.decoder = build_protocol_v11_decoder(output_shape=output_shape, base_channels=base_channels, vmin=vmin, vmax=vmax).module

    def to(self, device: str) -> "GeometryAwareTraceFeatureRegressor":
        self.decoder.to(device)
        for module in (
            self.bridge.trace_encoder.module,
            self.bridge.shot_encoder.module,
            self.bridge.condition_encoder.module,
            self.bridge.fusion.layer_norm,
            self.bridge.fusion.gate,
            self.bridge.fusion.gamma,
            self.bridge.fusion.beta,
        ):
            module.to(device)
        return self

    def __call__(self, *, trace_features: Any, records: Any, source_positions: Any, multiscale_feature: Any | None = None) -> Any:
        torch = require_torch_backend()
        g_trace_np, g_time_np = _canonical_geometry(records.detach().cpu().numpy(), source_positions.detach().cpu().numpy())
        g_trace = torch.as_tensor(g_trace_np, dtype=torch.float32, device=records.device).unsqueeze(3)
        g_time = torch.as_tensor(g_time_np, dtype=torch.float32, device=records.device)
        multiscale = None
        if multiscale_feature is not None:
            multiscale = multiscale_feature.unsqueeze(3)
        fused = self.bridge.forward(
            patch_tokens=trace_features.unsqueeze(3),
            records=records,
            g_trace=g_trace,
            g_time=g_time,
            multiscale_feature=multiscale,
        )["tokens"][:, :, :, 0, :]
        trace_feature = fused.mean(dim=2).mean(dim=1)
        return self.decoder(trace_feature)


def _extract_trace_features_for_method(*, method_key: str, bridge_name: str, records: np.ndarray, config: dict[str, Any], device: str, ncs_payload: dict[str, Any] | None = None) -> np.ndarray:
    torch = require_torch_backend()
    records_tensor = torch.as_tensor(records, dtype=torch.float32)
    batch, shots, receivers, samples = records_tensor.shape
    bridge = SeismicToVisionBridge(
        image_size=int(config["image_size"]),
        in_chans=3,
        norm_mode="zscore",
        feature_mode="raw_envelope_spectrum3",
    )
    chunk_size = int(config.get("trace_feature_chunk_size", 256))
    flat = records_tensor.reshape(batch * shots, receivers, samples).unsqueeze(1)
    outputs: list[np.ndarray] = []
    backbone = None
    if method_key == "dinov2_frozen":
        backbone = build_vision_backbone(
            backbone_type=str(config["backbones"]["dinov2_frozen"].get("backbone_type", "dummy")),
            model_name=str(config["backbones"]["dinov2_frozen"].get("model_name", "vit_tiny_patch16_224")),
            pretrained=False,
            image_size=int(config["image_size"]),
            in_chans=3,
            freeze=True,
        ).to(device)
    for start in range(0, flat.shape[0], chunk_size):
        shot_inputs = flat[start : start + chunk_size]
        images = bridge(shot_inputs).to(device)
        if method_key == "dinov2_frozen":
            with torch.no_grad():
                tokens = backbone(images)
            outputs.append(tokens.detach().cpu().numpy().astype(np.float32))
        else:
            if ncs_payload is None:
                raise RuntimeError("NCS payload is required for ncs2d_frozen")
            outputs.append(np.asarray(ncs_payload["model"].encode_tokens(images.detach().cpu().numpy()), dtype=np.float32))
    tokens = np.concatenate(outputs, axis=0)
    token_dim = int(tokens.shape[-1])
    token_count = int(tokens.shape[1])
    grid_h = int(round(token_count ** 0.5))
    while grid_h > 1 and token_count % grid_h != 0:
        grid_h -= 1
    grid_w = token_count // grid_h
    token_grid = tokens.reshape(batch, shots, grid_h, grid_w, token_dim)
    receiver_tokens = token_grid.mean(axis=2)
    receiver_index = np.linspace(0.0, grid_w - 1, receivers).round().astype(np.int64)
    features = receiver_tokens[:, :, receiver_index, :]
    return features.astype(np.float32)


def _cache_split_features(*, samples: list[dict[str, Any]], cache_path: Path, method_key: str, bridge_name: str, config: dict[str, Any], device: str, ncs_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=True) as payload:
            return {key: payload[key] for key in payload.files}
    records_list = []
    velocity_list = []
    source_positions_list = []
    sample_ids = []
    for row in samples:
        path = Path(str(row.get("path") or row.get("data_file")))
        sample = load_npz_sample(path)
        records_list.append(sample.records.astype(np.float32))
        velocity_list.append(sample.velocity.astype(np.float32))
        source_positions_list.append(sample.source_positions.astype(np.float32))
        sample_ids.append(str(row.get("sample_id") or path))
    records = np.stack(records_list, axis=0)
    features = _extract_trace_features_for_method(method_key=method_key, bridge_name=bridge_name, records=records, config=config, device=device, ncs_payload=ncs_payload)
    payload = {
        "records": records.astype(np.float32),
        "velocity": np.stack(velocity_list, axis=0)[:, None].astype(np.float32),
        "source_positions": np.stack(source_positions_list, axis=0).astype(np.float32),
        "trace_features": features.astype(np.float32),
        "sample_id": np.asarray(sample_ids, dtype=object),
        "multiscale_feature": _compute_multiscale_feature(records).astype(np.float32),
        "is_real_feature": np.asarray(bool(method_key == "ncs2d_frozen"), dtype=bool),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    return payload


def _evaluate_v14_model(model: GeometryAwareTraceFeatureRegressor, arrays: dict[str, np.ndarray], *, batch_size: int, device: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[str]]:
    from fwi_visionfm.evaluation.metrics import compute_velocity_metrics

    torch = require_torch_backend()
    predictions = []
    targets = []
    sample_ids: list[str] = []
    with torch.no_grad():
        for start in range(0, len(arrays["velocity"]), batch_size):
            end = start + batch_size
            pred = model(
                trace_features=torch.as_tensor(arrays["trace_features"][start:end], dtype=torch.float32, device=device),
                records=torch.as_tensor(arrays["records"][start:end], dtype=torch.float32, device=device),
                source_positions=torch.as_tensor(arrays["source_positions"][start:end], dtype=torch.float32, device=device),
                multiscale_feature=torch.as_tensor(arrays["multiscale_feature"][start:end], dtype=torch.float32, device=device),
            )
            predictions.append(pred.detach().cpu().numpy())
            targets.append(arrays["velocity"][start:end])
            sample_ids.extend([str(item) for item in arrays["sample_id"][start:end].tolist()])
    prediction = np.concatenate(predictions, axis=0).astype(np.float32)
    target = np.concatenate(targets, axis=0).astype(np.float32)
    metrics = compute_velocity_metrics(prediction, target)
    metrics["metric_space"] = "physical_velocity"
    return metrics, prediction, target, sample_ids


def _train_single_v14_run(*, run_dir: Path, row: dict[str, Any], manifest: dict[str, Any], config: dict[str, Any], device: str, cache_root: Path, ncs_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    torch = require_torch_backend()
    cache_arrays = {}
    for split, manifest_key in (
        ("train", "train_samples"),
        ("val", "val_samples"),
        ("in_family_test", "in_family_test_samples"),
        ("cross_family_test", "cross_family_test_samples"),
    ):
        cache_path = _trace_cache_dir(root=cache_root, transfer_id=row["transfer_id"], method_key=row["method_key"], seed=int(row["seed"])) / f"{split}.npz"
        cache_arrays[split] = _cache_split_features(
            samples=manifest[manifest_key],
            cache_path=cache_path,
            method_key=row["method_key"],
            bridge_name=row["bridge_name"],
            config=config,
            device=device,
            ncs_payload=ncs_payload,
        )
    token_dim = int(cache_arrays["train"]["trace_features"].shape[-1])
    vmin, vmax = _velocity_bounds(manifest)
    model = GeometryAwareTraceFeatureRegressor(
        token_dim=token_dim,
        output_shape=tuple(int(value) for value in config["velocity_shape"]),
        bridge_row=row,
        vmin=vmin,
        vmax=vmax,
        base_channels=int(config["decoder_base_channels"]),
        geometry_provenance=str(row["geometry_provenance"]),
    ).to(device)
    optimizer_report = build_geometry_bridge_optimizer_report(bridge=model.bridge, decoder=model.decoder, learning_rate=float(config["learning_rate"]))
    optimizer = optimizer_report.pop("optimizer")
    criterion = torch.nn.L1Loss()
    history = []
    train_arrays = cache_arrays["train"]
    seed = int(row["seed"])
    for epoch in range(1, int(config["epochs"]) + 1):
        order = np.random.default_rng(seed + epoch).permutation(len(train_arrays["velocity"]))
        losses = []
        for start in range(0, len(order), int(config["batch_size"])):
            index = order[start : start + int(config["batch_size"])]
            pred = model(
                trace_features=torch.as_tensor(train_arrays["trace_features"][index], dtype=torch.float32, device=device),
                records=torch.as_tensor(train_arrays["records"][index], dtype=torch.float32, device=device),
                source_positions=torch.as_tensor(train_arrays["source_positions"][index], dtype=torch.float32, device=device),
                multiscale_feature=torch.as_tensor(train_arrays["multiscale_feature"][index], dtype=torch.float32, device=device),
            )
            target = torch.as_tensor(train_arrays["velocity"][index], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "train_l1": float(np.mean(losses)) if losses else 0.0})
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_history(run_dir / "train_history.csv", history)
    results = {}
    for split, filename in (
        ("val", "metrics_val.json"),
        ("in_family_test", "metrics_in_family_test.json"),
        ("cross_family_test", "metrics_cross_family_test.json"),
    ):
        metrics, prediction, target, sample_ids = _evaluate_v14_model(model, cache_arrays[split], batch_size=int(config["batch_size"]), device=device)
        (run_dir / filename).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        if split != "val":
            write_protocol_v14_prediction_npz(
                run_dir / f"predictions_{split}.npz",
                prediction=prediction,
                target=target,
                sample_ids=sample_ids,
                metadata={
                    "model_id": row["method_id"],
                    "bridge_name": row["bridge_name"],
                    "geometry_mode": row["geometry_mode"],
                    "geometry_provenance": row["geometry_provenance"],
                    "trace_context_radius": row["trace_context_radius"],
                    "use_shot_global_context": row["use_shot_global_context"],
                    "use_multiscale_context": row["use_multiscale_context"],
                    "source_family": row["source_family"],
                    "target_family": row["target_family"],
                    "seed": int(row["seed"]),
                    "metric_space": row["metric_space"],
                    "is_real_feature": row["method_key"] == "ncs2d_frozen",
                },
            )
        results[split] = metrics
    (run_dir / "geometry_metadata.json").write_text(
        json.dumps(
            {
                "geometry_mode": row["geometry_mode"],
                "geometry_provenance": row["geometry_provenance"],
                "trace_context_radius": int(row["trace_context_radius"]),
                "use_shot_global_context": _bool(row["use_shot_global_context"]),
                "use_multiscale_context": _bool(row["use_multiscale_context"]),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "model_card.json").write_text(
        json.dumps(
            {
                "method_id": row["method_id"],
                "method_key": row["method_key"],
                "bridge_id": row["bridge_id"],
                "is_real_feature": row["method_key"] == "ncs2d_frozen",
                **optimizer_report,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_grid_png(run_dir / "prediction_grid.png", npz_path=run_dir / "predictions_cross_family_test.npz", gradient=False)
    _write_grid_png(run_dir / "gradient_grid.png", npz_path=run_dir / "predictions_cross_family_test.npz", gradient=True)
    return {"status": "SUCCESS", "metrics": results}


def run_protocol_v14_geometry_aware_trace_bridge(*, config_path: str | Path, output_dir: str | Path, stage: str, seeds: list[int], device: str, resume: bool) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    root = Path(output_dir)
    matrix_path = root / "protocol_v14_run_matrix.csv"
    if not matrix_path.exists():
        raise FileNotFoundError(f"run matrix not found: {matrix_path}")
    rows = _read_matrix(matrix_path)
    manifest_root = Path(str(config.get("manifest_root", root / "manifests")))
    manifests = _manifest_map(manifest_root)
    cache_root = root / "feature_cache"
    ncs_payload = None
    if any(row["method_key"] == "ncs2d_frozen" for row in rows):
        backbones = config.get("backbones", {})
        ncs_cfg = backbones.get("ncs2d_frozen", {})
        ncs_payload = load_ncs_model("ncs_2d", repo_path=ncs_cfg.get("ncs_repo"), weights_path=ncs_cfg.get("ncs_2d_weights"), device=device)
    completed = []
    for row in rows:
        seed = int(row["seed"])
        if seed not in seeds:
            continue
        if stage == "screening" and seed != 0:
            continue
        run_dir = root / "runs" / row["transfer_id"] / row["method_key"] / f"seed_{row['seed']}" / row["bridge_id"]
        if resume and _is_complete_v14_success(run_dir):
            completed.append({**row, "status": "SUCCESS", "reused": row["bridge_id"] == "B0"})
            continue
        if row["bridge_id"] == "B0" and row.get("status") == "REUSE_GATE_PASSED":
            source = Path(str(row["reused_from"]))
            if source.exists():
                _copy_reused_run(source, run_dir)
                completed.append({**row, "status": "SUCCESS", "reused": True})
                continue
        manifest = manifests.get((row["source_family"], row["target_family"], seed))
        if manifest is None:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "config.json").write_text(json.dumps({**row, "status": "SKIPPED_DATA_UNAVAILABLE"}, indent=2, ensure_ascii=False), encoding="utf-8")
            completed.append({**row, "status": "SKIPPED_DATA_UNAVAILABLE"})
            continue
        result = _train_single_v14_run(run_dir=run_dir, row=row, manifest=manifest, config=config, device=device, cache_root=cache_root, ncs_payload=ncs_payload)
        config_hash = _hash_payload({"protocol_config": config, "run": row})
        run_config = {**row, "status": result["status"], "manifest_combined_hash": manifest.get("manifest_combined_hash"), "locked_config_hash": config_hash, "target_test_used_for_training": False, "target_test_used_for_validation": False, "target_test_used_for_model_selection": False}
        (run_dir / "config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
        (run_dir / "config_hash.txt").write_text(config_hash + "\n", encoding="utf-8")
        (run_dir / "run_log.txt").write_text(f"status={result['status']}\n", encoding="utf-8")
        completed.append(run_config)
    payload = {
        "protocol": config.get("protocol", "protocol_v14_geometry_aware_trace_bridge"),
        "stage": stage,
        "requested_seeds": seeds,
        "run_count": len(completed),
        "success": sum(row["status"] == "SUCCESS" for row in completed),
        "failed": sum(str(row["status"]).startswith("FAILED") for row in completed),
        "skipped": sum(str(row["status"]).startswith("SKIPPED") for row in completed),
        "runs": completed,
    }
    (root / f"protocol_v14_{stage}_run_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", default="true")
    args = parser.parse_args()
    print(json.dumps(run_protocol_v14_geometry_aware_trace_bridge(config_path=args.config, output_dir=args.output_dir, stage=args.stage, seeds=args.seeds, device=args.device, resume=str(args.resume).lower() == "true"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
