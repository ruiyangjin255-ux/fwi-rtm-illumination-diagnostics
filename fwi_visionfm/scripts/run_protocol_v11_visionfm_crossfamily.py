# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import time
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from fwi_visionfm.datasets import load_npz_sample
from fwi_visionfm.evaluation.metrics import compute_velocity_metrics
from fwi_visionfm.models.protocol_v11_model_registry import build_protocol_v11_model, count_parameters, get_method_spec
from fwi_visionfm.models.seismic_bridge import SeismicToVisionBridge
from fwi_visionfm.models.seismic_backbones.ncs_backbone import load_ncs_model
from fwi_visionfm.torch_backend import require_torch_backend
from fwi_visionfm.torch_backend.data import build_torch_dataloader


REQUIRED_PREDICTION_FIELDS = {
    "velocity_pred_physical", "velocity_true_physical", "error_map_physical", "sample_id", "metric_space",
    "model_id", "bridge_name", "source_family", "target_family", "seed",
}

REQUIRED_SUCCESS_FILES = {
    "config.json",
    "model_card.json",
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

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _path_of(row: dict[str, Any]) -> str:
    return str(row.get("path") or row.get("data_file"))


def assert_target_test_isolation(manifest: dict[str, Any]) -> None:
    train_selection = {
        _path_of(row)
        for split in ("train_samples", "val_samples", "in_family_test_samples")
        for row in manifest[split]
    }
    target_test = {_path_of(row) for row in manifest["cross_family_test_samples"]}
    overlap = train_selection & target_test
    if overlap:
        raise ValueError(f"target cross-family test leakage detected: {sorted(overlap)[:3]}")
    source_family = manifest.get("source_family")
    target_family = manifest.get("target_family")
    if source_family and target_family and source_family == target_family:
        raise ValueError("cross-family manifest must use distinct source and target families")


def write_prediction_npz(
    path: str | Path,
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    sample_ids: list[str],
    metadata: dict[str, Any],
) -> Path:
    pred = np.asarray(prediction, dtype=np.float32)
    truth = np.asarray(target, dtype=np.float32)
    if pred.ndim == 3:
        pred = pred[:, None]
    if truth.ndim == 3:
        truth = truth[:, None]
    if pred.shape != truth.shape or pred.shape[-2:] != (70, 70):
        raise ValueError(f"Protocol V11 prediction contract requires matching [B,1,70,70], got {pred.shape} and {truth.shape}")
    if len(sample_ids) != pred.shape[0]:
        raise ValueError("sample_id count does not match prediction batch")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        velocity_pred_physical=pred,
        velocity_true_physical=truth,
        error_map_physical=np.abs(pred - truth).astype(np.float32),
        prediction=pred,
        target=truth,
        sample_id=np.asarray(sample_ids, dtype=str),
        metric_space=np.asarray("physical_velocity"),
        model_id=np.asarray(str(metadata["model_id"])),
        bridge_name=np.asarray(str(metadata["bridge_name"])),
        source_family=np.asarray(str(metadata["source_family"])),
        target_family=np.asarray(str(metadata["target_family"])),
        seed=np.asarray(int(metadata["seed"])),
    )
    return output


def _read_matrix(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any) -> bool:
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _manifest_map(root: Path) -> dict[tuple[str, str, int], Path]:
    tokens = {"flatvel_a": "flatvel", "curvevel_a": "curvevel", "flatfault_a": "flatfault"}
    result: dict[tuple[str, str, int], Path] = {}
    for path in sorted((root / "manifests").glob("*_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not {"source_family", "target_family", "seed"}.issubset(payload):
            continue
        source_name, target_name = payload["source_family"].lower(), payload["target_family"].lower()
        source = next((key for key, token in tokens.items() if token in source_name), None)
        target = next((key for key, token in tokens.items() if token in target_name), None)
        if source and target:
            result[(source, target, int(payload["seed"]))] = path
    return result


def _read_bounds(manifest: dict[str, Any]) -> tuple[float, float]:
    stats = json.loads(Path(manifest["stats_path"]).read_text(encoding="utf-8"))
    velocity = stats.get("velocity", {})
    return float(velocity.get("min", stats.get("target_min", 1500.0))), float(velocity.get("max", stats.get("target_max", 4500.0)))


def _paths(manifest: dict[str, Any], split: str) -> list[Path]:
    return [Path(_path_of(row)) for row in manifest[f"{split}_samples"]]


def _ids(paths: list[Path]) -> list[str]:
    return [f"{path.parent.name}/{path.name}" for path in paths]


def _to_contract_target(tensor: Any) -> Any:
    return tensor.unsqueeze(1) if tensor.ndim == 3 else tensor


def _evaluate_model(model: Any, loader: Any, *, device: str, shot_count: int) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[str]]:
    torch = require_torch_backend()
    module = getattr(model, "module", model)
    module.eval()
    predictions, targets, sample_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            records = batch["records"][:, :shot_count].to(device)
            positions = batch["source_positions"][:, :shot_count].to(device)
            target = _to_contract_target(batch["velocity"].to(device))
            prediction = module(records, positions)
            if prediction.ndim == 3:
                prediction = prediction.unsqueeze(1)
            predictions.append(prediction.detach().cpu().numpy())
            targets.append(target.detach().cpu().numpy())
            sample_ids.extend(f"{Path(path).parent.name}/{Path(path).name}" for path in batch["path"])
    pred = np.concatenate(predictions).astype(np.float32)
    truth = np.concatenate(targets).astype(np.float32)
    metrics = compute_velocity_metrics(pred, truth)
    metrics.update({"metric_space": "physical_velocity", "sample_count": int(len(pred))})
    return metrics, pred, truth, sample_ids


def _train_torch_model(
    *, model: Any, manifest: dict[str, Any], config: dict[str, Any], seed: int, device: str,
) -> tuple[list[dict[str, Any]], dict[str, tuple[dict[str, Any], np.ndarray, np.ndarray, list[str]]]]:
    torch = require_torch_backend()
    torch.manual_seed(seed)
    module = getattr(model, "module", model).to(device)
    batch_size = int(config["batch_size"])
    loaders = {
        split: build_torch_dataloader(_paths(manifest, split), batch_size=batch_size, shuffle=split == "train", seed=seed)
        for split in ("train", "val", "in_family_test", "cross_family_test")
    }
    parameters = [p for p in module.parameters() if p.requires_grad]
    if not parameters:
        raise RuntimeError("model has no trainable parameters")
    optimizer = torch.optim.Adam(parameters, lr=float(config["learning_rate"]))
    criterion = torch.nn.L1Loss()
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(config["epochs"]) + 1):
        module.train()
        losses = []
        for batch in loaders["train"]:
            records = batch["records"][:, : int(config["shot_count"])].to(device)
            positions = batch["source_positions"][:, : int(config["shot_count"])].to(device)
            target = _to_contract_target(batch["velocity"].to(device))
            optimizer.zero_grad(set_to_none=True)
            prediction = module(records, positions)
            if prediction.ndim == 3:
                prediction = prediction.unsqueeze(1)
            loss = criterion(prediction, target)
            loss.backward(); optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_metrics, _, _, _ = _evaluate_model(model, loaders["val"], device=device, shot_count=int(config["shot_count"]))
        history.append({"epoch": epoch, "train_l1": float(np.mean(losses)), "val_mae": val_metrics["mae"], "val_rmse": val_metrics["rmse"]})
    evaluations = {split: _evaluate_model(model, loaders[split], device=device, shot_count=int(config["shot_count"])) for split in ("val", "in_family_test", "cross_family_test")}
    return history, evaluations


def _extract_ncs_split(paths: list[Path], model_payload: dict[str, Any], bridge: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    torch = require_torch_backend()
    image_bridge = SeismicToVisionBridge(image_size=224, in_chans=3, norm_mode="zscore", feature_mode=bridge)
    features, targets = [], []
    for path in paths:
        sample = load_npz_sample(path)
        records = torch.as_tensor(sample.records[None, :5], dtype=torch.float32)
        images = image_bridge(records).detach().cpu().numpy()
        shot_features = model_payload["model"].encode(images)
        features.append(np.asarray(shot_features, dtype=np.float32).reshape(len(images), -1).mean(axis=0))
        targets.append(sample.velocity.astype(np.float32))
    return np.stack(features), np.stack(targets)[:, None], _ids(paths)


def _train_ncs_decoder(
    *, model: Any, manifest: dict[str, Any], config: dict[str, Any], seed: int, device: str, ncs_payload: dict[str, Any], cache_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, tuple[dict[str, Any], np.ndarray, np.ndarray, list[str]]]]:
    torch = require_torch_backend()
    torch.manual_seed(seed)
    module = model.module.to(device)
    arrays: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    cache_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "in_family_test", "cross_family_test"):
        cache = cache_dir / f"{split}.npz"
        if cache.is_file():
            with np.load(cache) as payload:
                arrays[split] = (payload["features"], payload["target"], payload["sample_id"].astype(str).tolist())
        else:
            arrays[split] = _extract_ncs_split(_paths(manifest, split), ncs_payload, "raw_envelope_spectrum3")
            np.savez_compressed(cache, features=arrays[split][0], target=arrays[split][1], sample_id=np.asarray(arrays[split][2], dtype=str), is_real_feature=np.asarray(True))
    optimizer = torch.optim.Adam(module.parameters(), lr=float(config["learning_rate"]))
    criterion = torch.nn.L1Loss()
    history = []
    features, target, _ = arrays["train"]
    for epoch in range(1, int(config["epochs"]) + 1):
        order = np.random.default_rng(seed + epoch).permutation(len(features))
        losses = []
        module.train()
        for start in range(0, len(order), int(config["batch_size"])):
            idx = order[start : start + int(config["batch_size"])]
            x = torch.as_tensor(features[idx], dtype=torch.float32, device=device)
            y = torch.as_tensor(target[idx], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True); pred = module(x); loss = criterion(pred, y); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "train_l1": float(np.mean(losses))})
    evaluations = {}
    module.eval()
    with torch.no_grad():
        for split in ("val", "in_family_test", "cross_family_test"):
            x, y, ids = arrays[split]
            parts = []
            for start in range(0, len(x), int(config["batch_size"])):
                parts.append(module(torch.as_tensor(x[start:start+int(config["batch_size"])], dtype=torch.float32, device=device)).cpu().numpy())
            pred = np.concatenate(parts).astype(np.float32)
            metrics = compute_velocity_metrics(pred, y); metrics.update({"metric_space": "physical_velocity", "sample_count": len(y), "is_real_feature": True})
            evaluations[split] = (metrics, pred, y, ids)
    return history, evaluations


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader(); writer.writerows(rows)


def _write_grid(path: Path, prediction: np.ndarray, target: np.ndarray, *, gradient: bool = False) -> None:
    count = min(4, len(prediction)); fig, axes = plt.subplots(count, 3, figsize=(10, 2.6 * count), squeeze=False)
    for index in range(count):
        truth, pred = target[index, 0], prediction[index, 0]
        if gradient:
            truth = np.hypot(*np.gradient(truth)); pred = np.hypot(*np.gradient(pred))
        error = np.abs(pred - truth)
        for ax, image, title in zip(axes[index], [truth, pred, error], ["真实", "预测", "绝对误差"]):
            ax.imshow(image, cmap="viridis" if not gradient else "magma"); ax.set_title(title); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _run_dir(root: Path, row: dict[str, str]) -> Path:
    return root / "runs" / row["transfer_id"] / row["method_key"] / f"seed_{row['seed']}"


def _is_complete_success(run_dir: Path) -> bool:
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if config.get("status") != "SUCCESS" or not all((run_dir / name).is_file() for name in REQUIRED_SUCCESS_FILES):
        return False
    try:
        model_card = json.loads((run_dir / "model_card.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    # Early V11 vision runs did not register the shared lazy decoder on the
    # trainable module. Their cards therefore omit about 2M decoder parameters.
    if model_card.get("kind") == "vision" and int(model_card.get("total_parameters", 0)) < 23_000_000:
        return False
    return True


def run_protocol_v11(*, config_path: str | Path, output_dir: str | Path, stage: str, seeds: list[int], device: str, resume: bool) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    matrix = _read_matrix(root / "protocol_v11_run_matrix.csv")
    manifest_map = _manifest_map(root)
    availability_path = root / "availability" / "protocol_v11_availability.json"
    availability = json.loads(availability_path.read_text(encoding="utf-8")) if availability_path.is_file() else {"backbones": []}
    backbone_status = {row["method_key"]: row["status"] for row in availability.get("backbones", [])}
    ncs_payload: dict[str, Any] | None = None
    completed = []
    for row in matrix:
        seed = int(row["seed"])
        if seed not in seeds or (stage == "screening" and seed != 0):
            continue
        run_dir = _run_dir(root, row); run_dir.mkdir(parents=True, exist_ok=True)
        if resume and _is_complete_success(run_dir):
            completed.append({**row, "status": "SUCCESS", "reused": True}); continue
        start = time.perf_counter(); status = row["status"]; reason = row.get("skip_reason", "")
        run_config = dict(row)
        try:
            if status == "SKIPPED_DATA_UNAVAILABLE":
                raise RuntimeError(reason)
            if backbone_status.get(row["method_key"], "AVAILABLE") != "AVAILABLE":
                status = "SKIPPED_BACKBONE_UNAVAILABLE"; reason = "backbone availability gate did not pass"
                raise RuntimeError(reason)
            key = (row["source_family"], row["target_family"], seed)
            manifest_path = manifest_map.get(key)
            if manifest_path is None:
                status = "SKIPPED_DATA_UNAVAILABLE"; reason = f"manifest unavailable for {key}"; raise RuntimeError(reason)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")); assert_target_test_isolation(manifest)
            vmin, vmax = _read_bounds(manifest); spec = get_method_spec(row["method_key"])
            model = build_protocol_v11_model(spec, config, vmin=vmin, vmax=vmax)
            if spec["kind"] == "ncs":
                if ncs_payload is None:
                    ncs_payload = load_ncs_model("ncs_2d", repo_path=config["backbones"].get("ncs_repo"), weights_path=config["backbones"].get("ncs_2d_weights"), device=device)
                if ncs_payload["status"] != "READY":
                    status = "SKIPPED_BACKBONE_UNAVAILABLE"; reason = f"NCS2D load status={ncs_payload['status']}"; raise RuntimeError(reason)
                history, evaluations = _train_ncs_decoder(model=model, manifest=manifest, config=config, seed=seed, device=device, ncs_payload=ncs_payload, cache_dir=root / "feature_cache" / row["transfer_id"] / f"seed_{seed}" / "ncs2d_mean_patch")
                is_real_feature = True
            else:
                history, evaluations = _train_torch_model(model=model, manifest=manifest, config=config, seed=seed, device=device)
                is_real_feature = False
            _write_history(run_dir / "train_history.csv", history)
            for split, filename in [("val", "metrics_val.json"), ("in_family_test", "metrics_in_family_test.json"), ("cross_family_test", "metrics_cross_family_test.json")]:
                (run_dir / filename).write_text(json.dumps(evaluations[split][0], indent=2, ensure_ascii=False), encoding="utf-8")
            metadata = {"model_id": row["method_id"], "bridge_name": row["bridge"], "source_family": row["source_family"], "target_family": row["target_family"], "seed": seed}
            write_prediction_npz(run_dir / "predictions_in_family_test.npz", prediction=evaluations["in_family_test"][1], target=evaluations["in_family_test"][2], sample_ids=evaluations["in_family_test"][3], metadata=metadata)
            write_prediction_npz(run_dir / "predictions_cross_family_test.npz", prediction=evaluations["cross_family_test"][1], target=evaluations["cross_family_test"][2], sample_ids=evaluations["cross_family_test"][3], metadata=metadata)
            _write_grid(run_dir / "prediction_grid.png", evaluations["cross_family_test"][1], evaluations["cross_family_test"][2])
            _write_grid(run_dir / "gradient_grid.png", evaluations["cross_family_test"][1], evaluations["cross_family_test"][2], gradient=True)
            parameters = count_parameters(model); status = "SUCCESS"; reason = ""
            model_card = {**spec, **parameters, "decoder": "common_bounded_velocity_decoder", "loss": "default_l1", "is_real_feature": is_real_feature}
            (run_dir / "model_card.json").write_text(json.dumps(model_card, indent=2, ensure_ascii=False), encoding="utf-8")
        except RuntimeError as exc:
            if status not in {"SKIPPED_DATA_UNAVAILABLE", "SKIPPED_BACKBONE_UNAVAILABLE"}:
                status = "FAILED"; reason = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            status = "FAILED"; reason = f"{type(exc).__name__}: {exc}"
            (run_dir / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        run_config.update({"status": status, "skip_reason": reason, "runtime_seconds": time.perf_counter() - start, "stage": stage, "device": device, "target_test_used_for_training": False, "target_test_used_for_model_selection": False})
        (run_dir / "config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")
        (run_dir / "run_log.txt").write_text(f"status={status}\nreason={reason}\nruntime_seconds={run_config['runtime_seconds']:.3f}\n", encoding="utf-8")
        completed.append(run_config)
    summary = {"stage": stage, "requested_seeds": seeds, "run_count": len(completed), "success": sum(r["status"] == "SUCCESS" for r in completed), "failed": sum(r["status"] == "FAILED" for r in completed), "skipped": sum(str(r["status"]).startswith("SKIPPED") for r in completed), "runs": completed}
    (root / f"protocol_v11_{stage}_run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--stage", choices=["screening", "full"], required=True); parser.add_argument("--seeds", type=int, nargs="+", required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--resume", default="true")
    args = parser.parse_args(); result = run_protocol_v11(config_path=args.config, output_dir=args.output_dir, stage=args.stage, seeds=args.seeds, device=args.device, resume=_bool(args.resume)); print(json.dumps({key: result[key] for key in ("stage", "run_count", "success", "failed", "skipped")}, indent=2))


if __name__ == "__main__": main()
