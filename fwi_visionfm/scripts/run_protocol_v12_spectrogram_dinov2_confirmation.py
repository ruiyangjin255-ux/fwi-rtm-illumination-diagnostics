# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fwi_visionfm.models.protocol_v11_model_registry import build_protocol_v11_model
from fwi_visionfm.torch_backend import require_torch_backend
from fwi_visionfm.torch_backend.data import build_torch_dataloader
try:
    from scripts.build_protocol_v12_manifests import combined_manifest_hash, compute_manifest_hashes
    from scripts.build_protocol_v12_matrix import V12_METHOD_SPECS
    from scripts.run_protocol_v11_visionfm_crossfamily import _evaluate_model, _paths, _read_bounds, _to_contract_target, _write_grid, _write_history, write_prediction_npz
except ModuleNotFoundError:  # direct script execution
    from build_protocol_v12_manifests import combined_manifest_hash, compute_manifest_hashes
    from build_protocol_v12_matrix import V12_METHOD_SPECS
    from run_protocol_v11_visionfm_crossfamily import _evaluate_model, _paths, _read_bounds, _to_contract_target, _write_grid, _write_history, write_prediction_npz


REQUIRED_SUCCESS_FILES = {
    "config.json", "config_hash.txt", "model_card.json", "train_history.csv",
    "metrics_val.json", "metrics_in_family_test.json", "metrics_cross_family_test.json",
    "predictions_in_family_test.npz", "predictions_cross_family_test.npz",
    "prediction_grid.png", "gradient_grid.png", "run_log.txt",
}


def _bool(value: Any) -> bool:
    return str(value).strip().lower() not in {"0", "false", "no", "off", ""}


def _parameter_count(parameters: list[Any]) -> int:
    from torch.nn.parameter import UninitializedParameter

    return int(sum(parameter.numel() for parameter in parameters if not isinstance(parameter, UninitializedParameter)))


def build_optimizer_with_registration_report(model: Any, *, learning_rate: float) -> tuple[Any, dict[str, Any]]:
    torch = require_torch_backend()
    module = getattr(model, "module", model)
    trainable = list(dict.fromkeys(parameter for parameter in module.parameters() if parameter.requires_grad))
    if not trainable:
        raise RuntimeError("model has no trainable parameters")
    optimizer = torch.optim.Adam(trainable, lr=float(learning_rate))
    decoder = getattr(module, "decoder", None)
    decoder_parameters = list(dict.fromkeys(decoder.parameters())) if decoder is not None else []
    decoder_trainable = [parameter for parameter in decoder_parameters if parameter.requires_grad]
    optimizer_parameters = list(dict.fromkeys(parameter for group in optimizer.param_groups for parameter in group["params"]))
    optimizer_ids = {id(parameter) for parameter in optimizer_parameters}
    decoder_optimizer = [parameter for parameter in decoder_trainable if id(parameter) in optimizer_ids]
    all_parameters = list(dict.fromkeys(module.parameters()))
    decoder_ids = {id(parameter) for parameter in decoder_parameters}
    encoder_parameters = [parameter for parameter in all_parameters if id(parameter) not in decoder_ids]
    report = {
        "encoder_parameters": _parameter_count(encoder_parameters),
        "decoder_parameters": _parameter_count(decoder_parameters),
        "trainable_parameters": _parameter_count(trainable),
        "optimizer_parameters": _parameter_count(optimizer_parameters),
        "decoder_optimizer_parameters": _parameter_count(decoder_optimizer),
        "decoder_fully_registered": {id(parameter) for parameter in decoder_trainable}.issubset(optimizer_ids),
        "total_parameters": _parameter_count(all_parameters),
        "uninitialized_decoder_parameter_objects": sum(type(parameter).__name__ == "UninitializedParameter" for parameter in decoder_parameters),
    }
    if not report["decoder_fully_registered"] or report["decoder_optimizer_parameters"] != _parameter_count(decoder_trainable):
        raise RuntimeError("common decoder parameters are not fully registered in optimizer")
    if report["optimizer_parameters"] != report["trainable_parameters"]:
        raise RuntimeError("optimizer parameter count differs from trainable parameter count")
    return optimizer, report


def assert_v12_target_isolation(manifest: dict[str, Any]) -> None:
    source_ids = {
        str(row.get("sample_id") or row.get("path"))
        for split in ("train_samples", "val_samples", "in_family_test_samples")
        for row in manifest[split]
    }
    target_ids = {str(row.get("sample_id") or row.get("path")) for row in manifest["cross_family_test_samples"]}
    if source_ids & target_ids:
        raise ValueError("target test leakage detected")
    if manifest.get("source_family") == manifest.get("target_family"):
        raise ValueError("cross-family source and target must differ")


def _train_model(*, model: Any, manifest: dict[str, Any], config: dict[str, Any], seed: int, device: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    torch = require_torch_backend(); torch.manual_seed(seed)
    module = getattr(model, "module", model).to(device)
    loaders = {split: build_torch_dataloader(_paths(manifest, split), batch_size=int(config["batch_size"]), shuffle=split == "train", seed=seed) for split in ("train", "val", "in_family_test", "cross_family_test")}
    first = next(iter(loaders["train"]))
    with torch.no_grad():
        module(first["records"][:, : int(config["shot_count"])].to(device), first["source_positions"][:, : int(config["shot_count"])].to(device))
    optimizer, parameter_report = build_optimizer_with_registration_report(module, learning_rate=float(config["learning_rate"]))
    criterion = torch.nn.L1Loss(); history = []
    for epoch in range(1, int(config["epochs"]) + 1):
        module.train(); losses = []
        for batch in loaders["train"]:
            records = batch["records"][:, : int(config["shot_count"])].to(device); positions = batch["source_positions"][:, : int(config["shot_count"])].to(device); target = _to_contract_target(batch["velocity"].to(device))
            optimizer.zero_grad(set_to_none=True); prediction = module(records, positions); prediction = prediction.unsqueeze(1) if prediction.ndim == 3 else prediction
            loss = criterion(prediction, target); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        val_metrics, _, _, _ = _evaluate_model(model, loaders["val"], device=device, shot_count=int(config["shot_count"]))
        history.append({"epoch": epoch, "train_l1": float(np.mean(losses)), "val_mae": val_metrics["mae"], "val_rmse": val_metrics["rmse"]})
    evaluations = {split: _evaluate_model(model, loaders[split], device=device, shot_count=int(config["shot_count"])) for split in ("val", "in_family_test", "cross_family_test")}
    return history, evaluations, parameter_report


def _canonical_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _manifest_map(manifest_dir: Path) -> dict[tuple[str, str, int], Path]:
    result = {}
    for path in manifest_dir.glob("*_manifest.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if {"source_family", "target_family", "seed"}.issubset(payload):
            result[(payload["source_family"], payload["target_family"], int(payload["seed"]))] = path
    return result


def _is_complete_success(run_dir: Path, expected_config_hash: str, expected_manifest_hash: str) -> bool:
    if not all((run_dir / name).is_file() for name in REQUIRED_SUCCESS_FILES):
        return False
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8")); card = json.loads((run_dir / "model_card.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return config.get("status") == "SUCCESS" and (run_dir / "config_hash.txt").read_text(encoding="utf-8").strip() == expected_config_hash and config.get("manifest_combined_hash") == expected_manifest_hash and card.get("decoder_fully_registered") is True and card.get("optimizer_parameters") == card.get("trainable_parameters")


def run_protocol_v12(*, config_path: str | Path, manifest_dir: str | Path, output_dir: str | Path, stage: str, seeds: list[int], device: str, resume: bool) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")); manifests = Path(manifest_dir); root = Path(output_dir); root.mkdir(parents=True, exist_ok=True)
    with (root / "run_matrix" / "protocol_v12_run_matrix.csv").open("r", encoding="utf-8", newline="") as handle: matrix = list(csv.DictReader(handle))
    recorded = json.loads((manifests / "protocol_v12_manifest_hashes.json").read_text(encoding="utf-8")); current_manifest_hash = combined_manifest_hash(compute_manifest_hashes(manifests))
    if current_manifest_hash != recorded["combined_hash"]:
        raise ValueError("manifest hash mismatch before training")
    manifest_map = _manifest_map(manifests); availability = json.loads((root / "availability" / "protocol_v12_availability.json").read_text(encoding="utf-8")); backbone_status = {row["method_key"]: row["status"] for row in availability["backbones"]}; specs = {row["method_key"]: row for row in V12_METHOD_SPECS}; completed = []
    for row in matrix:
        seed = int(row["seed"])
        if seed not in seeds or (stage == "screening" and seed != 0): continue
        run_dir = root / "runs" / row["transfer_id"] / row["method_key"] / f"seed_{seed}"; run_dir.mkdir(parents=True, exist_ok=True)
        locked_payload = {"protocol_config": config, "run": {key: value for key, value in row.items() if key not in {"status", "skip_reason"}}, "device": device}; config_hash = _canonical_hash(locked_payload)
        if resume and _is_complete_success(run_dir, config_hash, current_manifest_hash):
            completed.append({**row, "status": "SUCCESS", "reused": True}); continue
        start = time.perf_counter(); status = row["status"]; reason = row.get("skip_reason", ""); run_config = dict(row)
        try:
            if status == "SKIPPED_DATA_UNAVAILABLE": raise RuntimeError(reason)
            if backbone_status.get(row["method_key"], "AVAILABLE") != "AVAILABLE": status = "SKIPPED_BACKBONE_UNAVAILABLE"; reason = "backbone availability gate did not pass"; raise RuntimeError(reason)
            manifest_path = manifest_map.get((row["source_family"], row["target_family"], seed))
            if manifest_path is None: status = "SKIPPED_DATA_UNAVAILABLE"; reason = "locked transfer manifest unavailable"; raise RuntimeError(reason)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")); assert_v12_target_isolation(manifest)
            if manifest.get("manifest_combined_hash") != current_manifest_hash: status = "FAILED_MANIFEST_MISMATCH"; reason = "run manifest hash differs from locked hash"; raise RuntimeError(reason)
            spec = specs[row["method_key"]]; vmin, vmax = _read_bounds(manifest); model = build_protocol_v11_model(spec, config, vmin=vmin, vmax=vmax)
            history, evaluations, parameter_report = _train_model(model=model, manifest=manifest, config=config, seed=seed, device=device)
            if combined_manifest_hash(compute_manifest_hashes(manifests)) != current_manifest_hash: status = "FAILED_MANIFEST_MISMATCH"; reason = "manifest changed during training"; raise RuntimeError(reason)
            _write_history(run_dir / "train_history.csv", history)
            for split, filename in (("val", "metrics_val.json"), ("in_family_test", "metrics_in_family_test.json"), ("cross_family_test", "metrics_cross_family_test.json")):
                (run_dir / filename).write_text(json.dumps(evaluations[split][0], indent=2, ensure_ascii=False), encoding="utf-8")
            metadata = {"model_id": row["method_id"], "bridge_name": row["bridge"], "source_family": row["source_family"], "target_family": row["target_family"], "seed": seed}
            write_prediction_npz(run_dir / "predictions_in_family_test.npz", prediction=evaluations["in_family_test"][1], target=evaluations["in_family_test"][2], sample_ids=evaluations["in_family_test"][3], metadata=metadata)
            write_prediction_npz(run_dir / "predictions_cross_family_test.npz", prediction=evaluations["cross_family_test"][1], target=evaluations["cross_family_test"][2], sample_ids=evaluations["cross_family_test"][3], metadata=metadata)
            _write_grid(run_dir / "prediction_grid.png", evaluations["cross_family_test"][1], evaluations["cross_family_test"][2]); _write_grid(run_dir / "gradient_grid.png", evaluations["cross_family_test"][1], evaluations["cross_family_test"][2], gradient=True)
            card = {**spec, **parameter_report, "bridge": row["bridge"], "backbone": row["backbone"], "adapter": spec.get("adapter", "none"), "decoder": config["decoder"], "loss": config["loss"]}; (run_dir / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
            status = "SUCCESS"; reason = ""
        except RuntimeError as exc:
            if status not in {"SKIPPED_DATA_UNAVAILABLE", "SKIPPED_BACKBONE_UNAVAILABLE", "FAILED_MANIFEST_MISMATCH"}: status = "FAILED"; reason = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            status = "FAILED"; reason = f"{type(exc).__name__}: {exc}"; (run_dir / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        run_config.update({"status": status, "skip_reason": reason, "runtime_seconds": time.perf_counter() - start, "stage": stage, "device": device, "manifest_combined_hash": current_manifest_hash, "locked_config_hash": config_hash, "target_test_used_for_training": False, "target_test_used_for_validation": False, "target_test_used_for_model_selection": False})
        (run_dir / "config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8"); (run_dir / "config_hash.txt").write_text(config_hash + "\n", encoding="utf-8"); (run_dir / "run_log.txt").write_text(f"status={status}\nreason={reason}\nruntime_seconds={run_config['runtime_seconds']:.3f}\n", encoding="utf-8"); completed.append(run_config)
    summary = {"stage": stage, "run_count": len(completed), "success": sum(row["status"] == "SUCCESS" for row in completed), "failed": sum(str(row["status"]).startswith("FAILED") for row in completed), "skipped": sum(str(row["status"]).startswith("SKIPPED") for row in completed), "runs": completed}; (root / f"protocol_v12_{stage}_run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"); return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--manifest-dir", required=True); parser.add_argument("--output-dir", required=True); parser.add_argument("--stage", choices=["screening", "confirmation"], required=True); parser.add_argument("--seeds", type=int, nargs="+", required=True); parser.add_argument("--device", default="cpu"); parser.add_argument("--resume", default="true")
    args = parser.parse_args(); result = run_protocol_v12(config_path=args.config, manifest_dir=args.manifest_dir, output_dir=args.output_dir, stage=args.stage, seeds=args.seeds, device=args.device, resume=_bool(args.resume)); print(json.dumps({key: result[key] for key in ("stage", "run_count", "success", "failed", "skipped")}, indent=2))


if __name__ == "__main__":
    main()
