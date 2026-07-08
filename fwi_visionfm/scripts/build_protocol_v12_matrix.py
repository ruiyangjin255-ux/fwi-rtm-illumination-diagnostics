# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from fwi_visionfm.scripts.build_protocol_v2_splits import _compute_train_stats, _shape_payload
try:
    from scripts.build_protocol_v12_manifests import compute_manifest_hashes, combined_manifest_hash
    from scripts.check_protocol_v11_availability import check_backbone_availability, check_family_availability
except ModuleNotFoundError:  # direct script execution
    from build_protocol_v12_manifests import compute_manifest_hashes, combined_manifest_hash
    from check_protocol_v11_availability import check_backbone_availability, check_family_availability


TRANSFERS = [("flatvel_a", "curvevel_a"), ("flatvel_a", "flatfault_a"), ("curvevel_a", "flatfault_a")]
V12_METHOD_SPECS = [
    {"method_id": "M1", "method_key": "cnn_baseline", "method_name": "CNN baseline", "bridge": "raw_envelope_spectrum3", "pretraining_source": "none", "kind": "cnn", "transfer_mode": "scratch", "pretrained": False, "adapter": "none"},
    {"method_id": "M2", "method_key": "random_vit", "method_name": "random ViT", "bridge": "raw_envelope_spectrum3", "pretraining_source": "none", "kind": "vision", "transfer_mode": "scratch", "pretrained": False, "adapter": "none"},
    {"method_id": "M3", "method_key": "dinov2_frozen", "method_name": "DINOv2 frozen", "bridge": "raw_envelope_spectrum3", "pretraining_source": "natural_image_dinov2", "kind": "vision", "transfer_mode": "frozen", "pretrained": True, "adapter": "frozen"},
    {"method_id": "M4", "method_key": "dinov2_lora", "method_name": "DINOv2-LoRA", "bridge": "raw_envelope_spectrum3", "pretraining_source": "natural_image_dinov2", "kind": "vision", "transfer_mode": "lora", "pretrained": True, "adapter": "lora"},
    {"method_id": "M5", "method_key": "spectrogram_dinov2_lora", "method_name": "spectrogram-DINOv2-LoRA", "bridge": "spectrogram_multiband", "pretraining_source": "natural_image_dinov2", "kind": "vision", "transfer_mode": "lora", "pretrained": True, "adapter": "lora"},
]


def build_v12_matrix_rows(*, available_families: dict[str, bool], seeds: list[int], config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = config or {"train_size": 200, "val_size": 50, "in_family_test_size": 50, "cross_family_test_size": 50, "epochs": 2, "shot_count": 5, "image_size": 224, "metric_space": "physical_velocity", "decoder": "common_bounded_velocity_decoder", "loss": "default_l1", "aggregation": "mean"}
    rows = []
    for source, target in TRANSFERS:
        available = bool(available_families.get(source) and available_families.get(target))
        for seed in seeds:
            for spec in V12_METHOD_SPECS:
                rows.append({"run_id": f"{source}_to_{target}__{spec['method_key']}__seed{seed}", "transfer_id": f"{source}_to_{target}", "source_family": source, "target_family": target, **spec, "backbone": spec["method_key"], "decoder": cfg.get("decoder", "common_bounded_velocity_decoder"), "loss": cfg.get("loss", "default_l1"), "aggregation": cfg.get("aggregation", "mean"), "seed": int(seed), "train_size": int(cfg["train_size"]), "val_size": int(cfg["val_size"]), "in_family_test_size": int(cfg["in_family_test_size"]), "cross_family_test_size": int(cfg["cross_family_test_size"]), "epochs": int(cfg["epochs"]), "shot_count": int(cfg["shot_count"]), "image_size": int(cfg.get("image_size", 224)), "metric_space": cfg.get("metric_space", "physical_velocity"), "target_test_usage": "evaluation_only", "boundary_auxiliary": False, "geometry_embedding": False, "fusion": False, "status": "PENDING" if available else "SKIPPED_DATA_UNAVAILABLE", "skip_reason": "" if available else f"source or target family unavailable: {source}->{target}"})
    return rows


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["local_index"] = int(row["local_index"]); row["global_index"] = int(row["global_index"])
    return rows


def _pre_registered_spec() -> str:
    return """# Protocol V12 预注册确认性评测规范

本文件在正式训练前生成。协议固定为 CPU、200/50/50、2 epochs、seed=0/1/2、5 炮、70×70 physical velocity、mean aggregation、共同 bounded decoder、default L1；target test 仅用于最终评价。

## 预注册比较

- A: M5 spectrogram-DINOv2-LoRA vs M4 DINOv2-LoRA
- B: M5 vs M2 random ViT
- C: M5 vs M1 CNN baseline
- D: M3 DINOv2 frozen vs M2 random ViT
- E: M4 DINOv2-LoRA vs M3 DINOv2 frozen

## 判读

一致方向性证据要求至少两个 transfer 的 MAE/RMSE 改善、至少一个结构指标不差、至少 2/3 seed 同向、paired bootstrap MAE 95% CI 完全小于 0，并严格对齐 sample_id。否则按部分或混合证据、未形成一致证据判读。

不得写成已证明提升 FWI 泛化、已证明优于 CNN、benchmark winner 或工程应用级性能。
"""


def build_protocol_v12_matrix(*, repo_root: str | Path, config_path: str | Path, manifest_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    repo = Path(repo_root); config_file = Path(config_path) if Path(config_path).is_absolute() else repo / config_path; cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    manifests = Path(manifest_dir) if Path(manifest_dir).is_absolute() else repo / manifest_dir; out = Path(output_dir) if Path(output_dir).is_absolute() else repo / output_dir; out.mkdir(parents=True, exist_ok=True)
    current_hashes = compute_manifest_hashes(manifests); locked_hash = combined_manifest_hash(current_hashes)
    recorded = json.loads((manifests / "protocol_v12_manifest_hashes.json").read_text(encoding="utf-8"))
    if locked_hash != recorded["combined_hash"]:
        raise ValueError("manifest hash mismatch before matrix build")
    family_availability_rows = check_family_availability(cfg["data_root"], cfg["families"]); available = {row["family_key"]: row["status"] == "AVAILABLE" for row in family_availability_rows}
    backbone_rows = [row for row in check_backbone_availability(cfg) if row["method_key"] != "ncs2d_frozen"]
    availability_dir = out / "availability"; availability_dir.mkdir(exist_ok=True)
    (availability_dir / "protocol_v12_availability.json").write_text(json.dumps({"protocol": cfg["protocol"], "families": family_availability_rows, "backbones": backbone_rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    stats_paths: dict[str, Path] = {}
    for family in cfg["families"]:
        train = _read_csv(manifests / f"{family}_train200.csv")
        stats_path = manifests / f"{family}_train_stats.json"; _compute_train_stats(train, stats_path); stats_paths[family] = stats_path
    for source, target in TRANSFERS:
        train = _read_csv(manifests / f"{source}_train200.csv"); val = _read_csv(manifests / f"{source}_val50.csv"); in_test = _read_csv(manifests / f"{source}_test50.csv"); cross = _read_csv(manifests / f"{target}_test50.csv")
        for seed in cfg["seeds"]:
            payload = {"protocol": cfg["protocol"], "source_family": source, "target_family": target, "seed": int(seed), "manifest_combined_hash": locked_hash, "train_samples": train, "val_samples": val, "in_family_test_samples": in_test, "cross_family_test_samples": cross, "stats_path": str(stats_paths[source]), "input_shape": _shape_payload(train[0])[0], "velocity_shape": _shape_payload(train[0])[1]}
            (manifests / f"{source}_to_{target}_seed{seed}_manifest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    rows = build_v12_matrix_rows(available_families=available, seeds=list(cfg["seeds"]), config=cfg)
    matrix_dir = out / "run_matrix"; matrix_dir.mkdir(exist_ok=True); matrix_path = matrix_dir / "protocol_v12_run_matrix.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    pre = out / "pre_registered"; pre.mkdir(exist_ok=True); (pre / "protocol_v12_pre_registered_spec.md").write_text(_pre_registered_spec(), encoding="utf-8")
    payload = {"run_count": len(rows), "manifest_combined_hash": locked_hash, "rows": rows}; (matrix_dir / "protocol_v12_matrix.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo-root", required=True); parser.add_argument("--config", required=True); parser.add_argument("--manifest-dir", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); result = build_protocol_v12_matrix(repo_root=args.repo_root, config_path=args.config, manifest_dir=args.manifest_dir, output_dir=args.output_dir); print(f"run_count={result['run_count']}")


if __name__ == "__main__":
    main()
