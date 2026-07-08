# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from fwi_visionfm.datasets import load_npz_sample
from fwi_visionfm.models.protocol_v11_model_registry import METHOD_SPECS
from fwi_visionfm.models.seismic_bridge import SeismicToVisionBridge
from fwi_visionfm.scripts.build_protocol_v2_splits import build_protocol_v2_splits
from fwi_visionfm.torch_backend import require_torch_backend
try:
    from scripts.check_protocol_v11_availability import check_family_availability
except ModuleNotFoundError:  # direct script execution
    from check_protocol_v11_availability import check_family_availability


TRANSFERS = [("flatvel_a", "curvevel_a"), ("flatvel_a", "flatfault_a"), ("curvevel_a", "flatfault_a")]


def build_run_matrix_rows(*, available_families: dict[str, bool], seeds: list[int], config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cfg = config or {"train_size": 100, "val_size": 50, "in_family_test_size": 50, "cross_family_test_size": 50, "epochs": 2, "shot_count": 5, "image_size": 224, "metric_space": "physical_velocity", "decoder": "common_bounded_velocity_decoder", "loss": "default_l1", "aggregation": "mean"}
    rows = []
    for source, target in TRANSFERS:
        pair_available = bool(available_families.get(source) and available_families.get(target))
        for seed in seeds:
            for spec in METHOD_SPECS:
                rows.append({
                    "run_id": f"{source}_to_{target}__{spec['method_key']}__seed{seed}",
                    "transfer_id": f"{source}_to_{target}",
                    "source_family": source,
                    "target_family": target,
                    "method_id": spec["method_id"],
                    "method_key": spec["method_key"],
                    "method_name": spec["method_name"],
                    "bridge": spec["bridge"],
                    "backbone": spec["method_key"],
                    "pretraining_source": spec["pretraining_source"],
                    "decoder": cfg.get("decoder", "common_bounded_velocity_decoder"),
                    "loss": cfg.get("loss", "default_l1"),
                    "aggregation": cfg.get("aggregation", "mean"),
                    "seed": int(seed),
                    "train_size": int(cfg["train_size"]),
                    "val_size": int(cfg["val_size"]),
                    "in_family_test_size": int(cfg["in_family_test_size"]),
                    "cross_family_test_size": int(cfg["cross_family_test_size"]),
                    "epochs": int(cfg["epochs"]),
                    "shot_count": int(cfg["shot_count"]),
                    "image_size": int(cfg.get("image_size", 224)),
                    "metric_space": cfg.get("metric_space", "physical_velocity"),
                    "target_test_usage": "evaluation_only",
                    "boundary_auxiliary": False,
                    "geometry_embedding": False,
                    "fusion": False,
                    "status": "PENDING" if pair_available else "SKIPPED_DATA_UNAVAILABLE",
                    "skip_reason": "" if pair_available else f"source or target family unavailable: {source}->{target}",
                })
    return rows


def _pre_registered_spec(config: dict[str, Any]) -> str:
    return """# Protocol V11 预注册实验规范

## 1. 科学问题

视觉模型是否适用于多炮地震记录到二维速度模型的端到端 FWI，以及自然图像预训练、LoRA、频谱转换和地震域预训练在未见构造上的方向性差异。

## 2. 数据划分

源域仅使用 100/50/50 的训练、验证和同类型测试样本；目标域 50 个跨类型测试样本只用于最终评价，不参与模型选择、提前停止或超参数调整。

## 3. 模型矩阵

M1 CNN baseline；M2 random ViT；M3 DINOv2 frozen；M4 DINOv2-LoRA；M5 spectrogram-DINOv2-LoRA；M6 NCS2D frozen。

## 4. 固定训练配置

CPU，2 epochs，seed=0/1/2，5 炮，224×224 bridge 输入，70×70 速度输出，mean 聚合、共同 bounded decoder 和 default L1。主矩阵不使用 boundary auxiliary、geometry embedding 或 fusion。

## 5. 指标定义

物理速度空间 MAE、RMSE、SSIM、gradient_error、edge_MAE；同时报告同类型与跨类型误差差距。

## 6. 结论判读规则

- A 一致的跨构造方向性证据：至少两个可运行 transfer 同时优于 CNN 和 random ViT 的 MAE/RMSE，至少一个结构指标不劣，三个 seed 至少两个方向一致，配对 bootstrap MAE 差异置信区间不跨 0。
- B 部分或混合证据：数值与结构方向不一致、仅一个 transfer 改善、仅部分 seed 支持或只稳定优于一个基线。
- C 当前未形成一致证据：未稳定优于两个基线、transfer 方向不一致、seed 波动较大或 bootstrap 不稳定。

## 7. 不能做出的结论

不得写任一方法已经提升 FWI 泛化能力、已经在复杂 OOD 有效、已优于全部 CNN/FWI 基线、属于标准基准级验证结论或实际工程应用级性能。
"""


def _write_bridge_artifacts(out: Path, config: dict[str, Any]) -> None:
    manifests = sorted((out / "manifests").glob("*_manifest.json"))
    if not manifests:
        return
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    sample_path = Path(manifest["train_samples"][0]["path"])
    sample = load_npz_sample(sample_path)
    torch = require_torch_backend()
    records = torch.as_tensor(sample.records[None, : int(config["shot_count"])], dtype=torch.float32)
    examples = out / "bridge_examples"; examples.mkdir(parents=True, exist_ok=True)
    bridge_manifest = {"source_sample": str(sample_path), "shot_count": int(config["shot_count"]), "normalization": "zscore", "image_size": int(config["image_size"]), "bridges": {}}
    for name in ("raw_envelope_spectrum3", "spectrogram_multiband"):
        bridge_cfg = config["bridges"][name]
        bridge = SeismicToVisionBridge(
            image_size=int(config["image_size"]), in_chans=3, norm_mode=str(bridge_cfg["norm_mode"]), feature_mode=name,
            spectrogram_n_fft=int(bridge_cfg.get("n_fft", 64)), spectrogram_hop_length=int(bridge_cfg.get("hop_length", 16)),
            spectrogram_win_length=int(bridge_cfg.get("win_length", 64)), spectrogram_power=float(bridge_cfg.get("power", 1.0)),
        )
        images = bridge(records).detach().cpu().numpy()
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        for channel, ax in enumerate(axes):
            ax.imshow(images[0, channel], cmap="viridis"); ax.set_title(f"{name}\nchannel {channel + 1}"); ax.axis("off")
        fig.tight_layout(); figure_path = examples / f"{name}_example.png"; fig.savefig(figure_path, dpi=150); plt.close(fig)
        bridge_manifest["bridges"][name] = {**bridge_cfg, "example_path": str(figure_path), "output_shape": list(images.shape), "unintended_difference_guard": "M4/M5 share all settings except bridge_name"}
    (out / "manifests" / "protocol_v11_bridge_manifest.json").write_text(json.dumps(bridge_manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def build_protocol_v11_matrix(*, repo_root: str | Path, config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    repo = Path(repo_root)
    config_file = Path(config_path) if Path(config_path).is_absolute() else repo / config_path
    cfg = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    out = Path(output_dir) if Path(output_dir).is_absolute() else repo / output_dir
    out.mkdir(parents=True, exist_ok=True)
    availability = check_family_availability(cfg["data_root"], cfg.get("families"))
    available = {row["family_key"]: row["status"] == "AVAILABLE" for row in availability}
    build_protocol_v2_splits(data_root=cfg["data_root"], output_root=out, train_size=cfg["train_size"], val_size=cfg["val_size"], test_size=cfg["in_family_test_size"], seeds=cfg["seeds"])
    _write_bridge_artifacts(out, cfg)
    rows = build_run_matrix_rows(available_families=available, seeds=list(cfg["seeds"]), config=cfg)
    matrix_path = out / "protocol_v11_run_matrix.csv"
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (out / "protocol_v11_pre_registered_spec.md").write_text(_pre_registered_spec(cfg), encoding="utf-8")
    payload = {"run_count": len(rows), "available_family_count": sum(available.values()), "rows": rows}
    (out / "protocol_v11_matrix.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True); parser.add_argument("--config", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = build_protocol_v11_matrix(repo_root=args.repo_root, config_path=args.config, output_dir=args.output_dir)
    print(f"run_count={result['run_count']}")


if __name__ == "__main__": main()
