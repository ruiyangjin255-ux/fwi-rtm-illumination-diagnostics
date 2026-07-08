# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


CONTRACT_FIELDS = (
    "manifest_hash", "sample_ids_hash", "source_family", "target_family", "seed",
    "shot_count", "bridge", "image_size", "decoder", "decoder_config_hash",
    "optimizer_registered", "loss", "epochs", "metric_space", "target_isolated",
)


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def verify_manifest_hash(files: dict[str, Path], expected: str | None) -> str:
    hashes = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in sorted(files.items())}
    current = _canonical_hash(hashes)
    if expected is not None and current != expected:
        raise ValueError("manifest SHA256 mismatch")
    return current


def compare_reuse_contract(v12: dict[str, Any], v13: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = [field for field in CONTRACT_FIELDS if v12.get(field) != v13.get(field)]
    return not reasons, reasons


def _decoder_hash(config: dict[str, Any]) -> str:
    return _canonical_hash({key: config[key] for key in ("velocity_shape", "decoder", "decoder_base_channels", "loss", "batch_size", "learning_rate", "aggregation")})


def _sample_ids_hash(manifest: dict[str, Any]) -> str:
    return _canonical_hash({split: [str(row.get("sample_id") or row.get("path")) for row in manifest[f"{split}_samples"]] for split in ("train", "val", "in_family_test", "cross_family_test")})


def _expected_contract(config: dict[str, Any], run: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    return {"manifest_hash": manifest["manifest_combined_hash"], "sample_ids_hash": _sample_ids_hash(manifest), "source_family": run["source_family"], "target_family": run["target_family"], "seed": int(run["seed"]), "shot_count": int(config["shot_count"]), "bridge": run["bridge"], "image_size": int(config["image_size"]), "decoder": config["decoder"], "decoder_config_hash": _decoder_hash(config), "optimizer_registered": True, "loss": config["loss"], "epochs": int(config["epochs"]), "metric_space": config["metric_space"], "target_isolated": True}


def _actual_contract(v12_config: dict[str, Any], v12_run: dict[str, Any], card: dict[str, Any], manifest: dict[str, Any], prediction_path: Path) -> dict[str, Any]:
    with np.load(prediction_path) as payload:
        prediction_ids = payload["sample_id"].astype(str).tolist()
    expected_cross = [str(row.get("sample_id") or row.get("path")) for row in manifest["cross_family_test_samples"]]
    target_isolated = not any(v12_run.get(key) for key in ("target_test_used_for_training", "target_test_used_for_validation", "target_test_used_for_model_selection")) and prediction_ids == expected_cross
    return {"manifest_hash": v12_run.get("manifest_combined_hash"), "sample_ids_hash": _sample_ids_hash(manifest), "source_family": v12_run["source_family"], "target_family": v12_run["target_family"], "seed": int(v12_run["seed"]), "shot_count": int(v12_run["shot_count"]), "bridge": v12_run["bridge"], "image_size": int(v12_run["image_size"]), "decoder": v12_run["decoder"], "decoder_config_hash": _decoder_hash(v12_config), "optimizer_registered": card.get("decoder_fully_registered") is True and card.get("optimizer_parameters") == card.get("trainable_parameters"), "loss": v12_run["loss"], "epochs": int(v12_run["epochs"]), "metric_space": v12_run["metric_space"], "target_isolated": target_isolated}


def verify_v12_reuse(*, v12_root: str | Path, config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    root = Path(v12_root); config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")); v12_config = yaml.safe_load((Path(__file__).parents[1] / "configs" / "protocol_v12_spectrogram_dinov2_confirmation.yaml").read_text(encoding="utf-8")); out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    manifest_map = {}
    for path in (root / "manifests").glob("*_manifest.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if {"source_family", "target_family", "seed"}.issubset(data): manifest_map[(data["source_family"], data["target_family"], int(data["seed"]))] = (path, data)
    rows = []
    for config_file in sorted(root.glob("runs/*/*/seed_*/config.json")):
        run = json.loads(config_file.read_text(encoding="utf-8"))
        if run["method_key"] == "ncs2d_frozen": continue
        card_path = config_file.parent / "model_card.json"; prediction = config_file.parent / "predictions_cross_family_test.npz"; key = (run["source_family"], run["target_family"], int(run["seed"])); reasons = []
        if run.get("status") != "SUCCESS": reasons.append("status")
        if key not in manifest_map: reasons.append("manifest_missing")
        if not card_path.is_file() or not prediction.is_file(): reasons.append("required_output_missing")
        actual = expected = {}
        if not reasons:
            _, manifest = manifest_map[key]; card = json.loads(card_path.read_text(encoding="utf-8")); actual = _actual_contract(v12_config, run, card, manifest, prediction); expected = _expected_contract(config, run, manifest); passed, mismatch = compare_reuse_contract(actual, expected); reasons.extend(mismatch)
        rows.append({"run_id": run["run_id"], "method_key": run["method_key"], "transfer_id": run["transfer_id"], "seed": int(run["seed"]), "source_run_dir": str(config_file.parent), "reusable": not reasons, "reasons": reasons, "actual_contract": actual, "expected_contract": expected})
    reusable = [row for row in rows if row["reusable"]]; nonreusable = [row for row in rows if not row["reusable"]]
    payload = {"v12_root": str(root), "total_runs": len(rows), "reusable_count": len(reusable), "nonreusable_count": len(nonreusable), "all_reusable": len(rows) == 45 and not nonreusable, "rows": rows}
    (out / "v12_reuse_verification.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["run_id", "method_key", "transfer_id", "seed", "source_run_dir", "reusable", "reasons"]
    for name, selected in (("v12_reusable_runs.csv", reusable), ("v12_nonreusable_runs.csv", nonreusable)):
        with (out / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows([{**row, "reasons": ";".join(row["reasons"])} for row in selected])
    lines = ["# V12 → V13 严格复用门禁", "", f"- V12 run 总数：{len(rows)}", f"- 可复用：{len(reusable)}", f"- 不可复用：{len(nonreusable)}", f"- 全部通过：{payload['all_reusable']}", "", "门禁逐 run 核验 manifest、sample_id、family、seed、shot、bridge、输入尺寸、decoder、optimizer、loss、epochs、metric space 与 target test 隔离。", ""]
    (out / "v12_reuse_verification.md").write_text("\n".join(lines), encoding="utf-8"); return payload


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--v12-root", required=True); parser.add_argument("--config", required=True); parser.add_argument("--output-dir", required=True); args = parser.parse_args(); print(json.dumps({key: value for key, value in verify_v12_reuse(v12_root=args.v12_root, config_path=args.config, output_dir=args.output_dir).items() if key != "rows"}, indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
