# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

from fwi_visionfm.models.seismic_backbones.ncs_backbone import get_ncs_status


TRANSFERS = (("flatvel_a", "curvevel_a"), ("flatvel_a", "flatfault_a"), ("curvevel_a", "flatfault_a"))
METHODS = (
    ("M1", "cnn_baseline", "CNN baseline", "raw_envelope_spectrum3", "none", "none"),
    ("M2", "random_vit", "random ViT", "raw_envelope_spectrum3", "none", "none"),
    ("M3", "dinov2_frozen", "DINOv2 frozen", "raw_envelope_spectrum3", "natural_image_dinov2", "frozen"),
    ("M4", "dinov2_lora", "DINOv2-LoRA", "raw_envelope_spectrum3", "natural_image_dinov2", "lora"),
    ("M5", "spectrogram_dinov2_lora", "spectrogram-DINOv2-LoRA", "spectrogram_multiband", "natural_image_dinov2", "lora"),
    ("M6", "ncs2d_frozen", "NCS2D frozen", "raw_envelope_spectrum3", "seismic_ncs2d", "frozen"),
)


def build_v13_matrix_rows(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = config or {"train_size": 200, "val_size": 50, "in_family_test_size": 50, "cross_family_test_size": 50, "epochs": 2, "shot_count": 5, "image_size": 224, "metric_space": "physical_velocity", "decoder": "common_bounded_velocity_decoder", "loss": "default_l1", "aggregation": "mean", "seeds": [0,1,2]}
    rows = []
    for source, target in TRANSFERS:
        for seed in cfg["seeds"]:
            for method_id, key, name, bridge, pretraining, adapter in METHODS:
                rows.append({"run_id": f"{source}_to_{target}__{key}__seed{seed}", "transfer_id": f"{source}_to_{target}", "source_family": source, "target_family": target, "method_id": method_id, "method_key": key, "method_name": name, "bridge": bridge, "pretraining_source": pretraining, "adapter": adapter, "backbone": key, "decoder": cfg["decoder"], "loss": cfg["loss"], "aggregation": cfg["aggregation"], "seed": int(seed), "train_size": int(cfg["train_size"]), "val_size": int(cfg["val_size"]), "in_family_test_size": int(cfg["in_family_test_size"]), "cross_family_test_size": int(cfg["cross_family_test_size"]), "epochs": int(cfg["epochs"]), "shot_count": int(cfg["shot_count"]), "image_size": int(cfg["image_size"]), "metric_space": cfg["metric_space"], "boundary_auxiliary": False, "geometry_embedding": False, "fusion": False, "status": "PENDING" if key == "ncs2d_frozen" else "REUSE_GATE_REQUIRED"})
    return rows


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists(): return
    try: os.link(source, target)
    except OSError: shutil.copy2(source, target)


def _copy_reused_run(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if not path.is_file(): continue
        if path.name == "config.json":
            config = json.loads(path.read_text(encoding="utf-8")); config["reused_from"] = "protocol_v12"; config["source_run_dir"] = str(source); (target / path.name).write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        else: _link_or_copy(path, target / path.name)


def _pre_registered() -> str:
    return """# Protocol V13 预注册规范

本文件在任何 M6 训练前生成。协议固定为 CPU、200/50/50、三 transfer、seed=0/1/2、2 epochs、5 炮、70×70 physical velocity、mean aggregation、共同 bounded decoder 和 default L1。主矩阵不含 boundary auxiliary、geometry、fusion 或 physics loss。

## 核心比较

- A: M3 DINOv2 frozen vs M2 random ViT
- B: M6 NCS2D frozen vs M2 random ViT
- C: M6 NCS2D frozen vs M3 DINOv2 frozen
- D: M5 spectrogram-DINOv2-LoRA vs M6 NCS2D frozen
- E: M4 DINOv2-LoRA vs M3 DINOv2 frozen

一致方向性证据要求至少 2/3 transfer 的 MAE/RMSE 更低、至少一个结构指标不更差、每个满足 transfer 至少 2/3 seed 同向，且 MAE paired bootstrap CI<0 至少 2/3 seed。不得表述为已证明提升泛化或预训练来源优胜。
"""


def build_protocol_v13_matrix(*, repo_root: str | Path, config_path: str | Path, reuse_gate_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    repo = Path(repo_root); config_file = Path(config_path) if Path(config_path).is_absolute() else repo / config_path; config = yaml.safe_load(config_file.read_text(encoding="utf-8")); gate = json.loads(Path(reuse_gate_path).read_text(encoding="utf-8")); out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = build_v13_matrix_rows(config); reusable = {row["run_id"]: row for row in gate["rows"] if row["reusable"]}
    for row in rows:
        if row["method_key"] == "ncs2d_frozen": continue
        gate_row = reusable.get(row["run_id"])
        if gate_row:
            _copy_reused_run(Path(gate_row["source_run_dir"]), out / "runs" / row["transfer_id"] / row["method_key"] / f"seed_{row['seed']}"); row["status"] = "SUCCESS"; row["reused_from"] = "protocol_v12"
        else: row["status"] = "REQUIRES_RERUN"; row["reused_from"] = ""
    v12_manifest = Path(config["manifest_root"]); v13_manifest = out / "manifests"; v13_manifest.mkdir(exist_ok=True)
    for path in v12_manifest.iterdir():
        if path.is_file(): _link_or_copy(path, v13_manifest / path.name)
    ncs = get_ncs_status("ncs_2d", repo_path=config["backbones"]["ncs_repo"], weights_path=config["backbones"]["ncs_2d_weights"]); availability = out / "availability"; availability.mkdir(exist_ok=True); (availability / "protocol_v13_availability.json").write_text(json.dumps({"ncs2d": ncs}, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "protocol_v13_run_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = sorted({key for row in rows for key in row}); writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    (out / "protocol_v13_pre_registered_spec.md").write_text(_pre_registered(), encoding="utf-8"); payload = {"run_count": len(rows), "reused_count": sum(row.get("reused_from") == "protocol_v12" for row in rows), "ncs_pending_count": sum(row["method_key"] == "ncs2d_frozen" for row in rows), "nonreusable_count": sum(row["status"] == "REQUIRES_RERUN" for row in rows), "rows": rows}; (out / "protocol_v13_matrix.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"); return payload


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo-root", required=True); parser.add_argument("--config", required=True); parser.add_argument("--v12-reuse-gate", required=True); parser.add_argument("--output-dir", required=True); args = parser.parse_args(); result = build_protocol_v13_matrix(repo_root=args.repo_root, config_path=args.config, reuse_gate_path=args.v12_reuse_gate, output_dir=args.output_dir); print(json.dumps({key:value for key,value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__": main()
