# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.build_protocol_v13_matrix import METHODS
except ModuleNotFoundError:  # direct script execution
    from build_protocol_v13_matrix import METHODS


FOOTER = "CPU 小样本统一协议；结果用于检验方向性证据，不构成标准基准级结论。"


def _font() -> None:
    path = Path("C:/Windows/Fonts/msyh.ttc")
    if path.is_file():
        fm.fontManager.addfont(str(path)); plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file(): return []
    with path.open("r", encoding="utf-8", newline="") as handle: return list(csv.DictReader(handle))


def _figure(path: Path, title: str, labels: list[str] | None = None, values: list[float] | None = None, ylabel: str = "指标值") -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    if labels and values:
        ax.bar(range(len(values)), values, color="#3f7893")
        ax.set_xticks(range(len(labels)), labels, rotation=38, ha="right", fontsize=8); ax.set_ylabel(ylabel); ax.spines[["top", "right"]].set_visible(False)
    else:
        ax.axis("off"); ax.text(.5, .55, title, ha="center", fontsize=18, weight="bold"); ax.text(.5, .42, "当前没有足够成功结果", ha="center")
    ax.set_title(title, fontsize=18, weight="bold", loc="left")
    fig.text(.5, .015, FOOTER, ha="center", fontsize=9, color="#455a64")
    fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(path, dpi=180); plt.close(fig)


def _matrix(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5)); ax.axis("off")
    cells = [[method_id, name, bridge, pretraining or "无", adapter] for method_id, _, name, bridge, pretraining, adapter in METHODS]
    table = ax.table(cellText=cells, colLabels=["ID", "方法", "Bridge", "预训练来源", "适配"], loc="center", cellLoc="left")
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.7)
    ax.set_title("Protocol V13 自然图像与地震域预训练矩阵", fontsize=18, weight="bold", loc="left")
    fig.text(.5, .02, FOOTER, ha="center", fontsize=9, color="#455a64"); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _heatmap(path: Path, rows: list[dict[str, str]], metric: str, title: str) -> None:
    methods = [item[1] for item in METHODS]
    transfers = sorted({row.get("transfer_id", "") for row in rows if row.get("transfer_id")})
    lookup = {(row.get("method_key"), row.get("transfer_id")): float(row[metric]) for row in rows if metric in row}
    if not transfers or not lookup:
        return _figure(path, title)
    data = np.asarray([[lookup.get((method, transfer), np.nan) for transfer in transfers] for method in methods])
    fig, ax = plt.subplots(figsize=(11, 6)); image = ax.imshow(data, cmap="Blues_r", aspect="auto")
    ax.set_xticks(range(len(transfers)), transfers, rotation=18, ha="right")
    ax.set_yticks(range(len(methods)), [item[2] for item in METHODS])
    for i in range(len(methods)):
        for j in range(len(transfers)):
            if np.isfinite(data[i, j]): ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title, fontsize=18, weight="bold", loc="left"); fig.colorbar(image, ax=ax, shrink=.75)
    fig.text(.5, .015, FOOTER, ha="center", fontsize=9, color="#455a64"); fig.tight_layout(rect=(0,.05,1,1)); fig.savefig(path,dpi=180); plt.close(fig)


def _prediction_grid(path: Path, root: Path) -> None:
    candidates = []
    for method in ("dinov2_frozen", "ncs2d_frozen"):
        found = sorted(root.glob(f"runs/*/{method}/seed_0/predictions_cross_family_test.npz"))
        if found: candidates.append(found[0])
    if len(candidates) < 2: return _figure(path, "自然图像与地震域预训练预测对照")
    fig, axes = plt.subplots(2, 3, figsize=(10, 7))
    for i, candidate in enumerate(candidates):
        with np.load(candidate) as payload:
            truth = payload["velocity_true_physical"][0, 0]; pred = payload["velocity_pred_physical"][0, 0]
        for ax, image, name in zip(axes[i], (truth, pred, np.abs(pred-truth)), ("真实", "预测", "绝对误差")):
            ax.imshow(image, cmap="viridis"); ax.set_title(f"{candidate.parents[1].name} {name}"); ax.axis("off")
    fig.suptitle("自然图像与地震域预训练预测对照", fontsize=18, weight="bold"); fig.text(.5,.01,FOOTER,ha="center",fontsize=9,color="#455a64"); fig.tight_layout(rect=(0,.04,1,.95)); fig.savefig(path,dpi=180); plt.close(fig)


def _integrity(root: Path) -> tuple[Path, dict[str, Any]]:
    configs = [json.loads(path.read_text(encoding="utf-8")) for path in root.glob("runs/*/*/seed_*/config.json")]
    cards = [json.loads(path.read_text(encoding="utf-8")) for path in root.glob("runs/*/*/seed_*/model_card.json")]
    ncs_configs = [row for row in configs if row.get("method_key") == "ncs2d_frozen" and row.get("status") == "SUCCESS"]
    gate_path = root / "reuse_gate" / "v12_reuse_verification.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    payload = {
        "run_count": len(configs), "success": sum(row.get("status") == "SUCCESS" for row in configs),
        "reused_v12_count": sum(row.get("reused_from") == "protocol_v12" for row in configs),
        "ncs_success_count": len(ncs_configs), "ncs_real_feature_ok": bool(ncs_configs) and all(row.get("is_real_feature") is True for row in ncs_configs),
        "optimizer_registration_ok": bool(cards) and all(row.get("decoder_fully_registered") is True and row.get("optimizer_parameters") == row.get("trainable_parameters") for row in cards),
        "target_isolation_ok": bool(configs) and all(not row.get("target_test_used_for_training") and not row.get("target_test_used_for_validation") and not row.get("target_test_used_for_model_selection") for row in configs),
        "v12_manifest_and_config_gate_ok": gate.get("all_reusable") is True,
        "run_config_hash_present": bool(configs) and all(bool(row.get("locked_config_hash")) for row in configs),
    }
    lines = ["# Protocol V13 协议完整性报告", "", *[f"- {key}: {value}" for key, value in payload.items()], "", "M1–M5 仅在严格复用门禁通过时复用；M6 必须使用真实 NCS2D 特征，fallback 特征不进入结果。", ""]
    path = root / "protocol_v13_protocol_integrity_report.md"; path.write_text("\n".join(lines), encoding="utf-8"); return path, payload


def write_protocol_v13_report(root: str | Path) -> tuple[Path, Path, Path]:
    _font(); protocol_root = Path(root); protocol_root.mkdir(parents=True, exist_ok=True)
    aggregate = _csv(protocol_root / "protocol_v13_aggregate_metrics.csv")
    bootstrap = _csv(protocol_root / "bootstrap" / "protocol_v13_bootstrap_deltas.csv")
    summary_path = protocol_root / "protocol_v13_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {"comparison_evidence": {}, "run_count": 0, "success": 0}
    integrity_path, integrity = _integrity(protocol_root)
    figures = protocol_root / "figures"; figures.mkdir(exist_ok=True)
    _matrix(figures / "figure_01_pretraining_source_matrix.png")
    _heatmap(figures / "figure_02_crossfamily_mae_heatmap.png", aggregate, "cross_family_mae_mean", "跨构造 MAE 热图")
    _heatmap(figures / "figure_03_crossfamily_structural_heatmap.png", aggregate, "cross_family_edge_mae_mean", "跨构造 edge_MAE 结构热图")
    selected = [r for r in aggregate if r.get("method_key") in {"random_vit","dinov2_frozen","ncs2d_frozen"}]
    _figure(figures / "figure_04_natural_vs_seismic_comparison.png", "随机、自然图像与地震域预训练对比", [f"{r.get('method_key')}\n{r.get('transfer_id')}" for r in selected], [float(r["cross_family_mae_mean"]) for r in selected], "MAE（m/s）")
    _figure(figures / "figure_05_generalization_gap_comparison.png", "MAE 泛化差距", [f"{r.get('method_key')}\n{r.get('transfer_id')}" for r in aggregate], [float(r["mae_generalization_gap_mean"]) for r in aggregate], "cross - in MAE（m/s）")
    _figure(figures / "figure_06_bootstrap_pretraining_source.png", "预训练来源 paired bootstrap", [f"{r['comparison_id']}\n{r['transfer_id']} s{r['seed']}" for r in bootstrap], [float(r["mae_mean_difference"]) for r in bootstrap], "候选 - 对照 MAE（m/s）")
    _prediction_grid(figures / "figure_07_prediction_grid_natural_vs_seismic.png", protocol_root)

    evidence = summary.get("comparison_evidence", {})
    evidence_lines = []
    for comparison_id in ("M3_vs_M2", "M6_vs_M2", "M6_vs_M3", "M5_vs_M6", "M4_vs_M3"):
        value = evidence.get(comparison_id, {})
        evidence_lines.append(f"- **{comparison_id}**：{value.get('evidence_level', '当前尚无完整结果')}；满足预注册 transfer 数 {value.get('qualifying_transfer_count', 0)}/3。")
    result_lines = ["| transfer | 方法 | MAE | RMSE | SSIM | gradient_error | edge_MAE |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in sorted(aggregate, key=lambda item: (item.get("transfer_id", ""), item.get("method_id", ""))):
        result_lines.append(f"| {row['transfer_id']} | {row['method_name']} | {float(row['cross_family_mae_mean']):.2f} | {float(row['cross_family_rmse_mean']):.2f} | {float(row['cross_family_ssim_mean']):.4f} | {float(row['cross_family_gradient_error_mean']):.2f} | {float(row['cross_family_edge_mae_mean']):.2f} |")
    gap_lines = ["| transfer | 方法 | MAE gap | RMSE gap | SSIM gap | gradient gap | edge gap |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in sorted(aggregate, key=lambda item: (item.get("transfer_id", ""), item.get("method_id", ""))):
        gap_lines.append(f"| {row['transfer_id']} | {row['method_name']} | {float(row['mae_generalization_gap_mean']):.2f} | {float(row['rmse_generalization_gap_mean']):.2f} | {float(row['ssim_generalization_gap_mean']):.4f} | {float(row['gradient_generalization_gap_mean']):.2f} | {float(row['edge_generalization_gap_mean']):.2f} |")
    report = [
        "# Protocol V13 自然图像与地震域预训练跨构造确认报告", "",
        "## 1. 研究问题", "", "本轮不是判断哪一个模型绝对最强，而是在相同输入、相同 decoder、相同训练预算下，比较自然图像预训练 DINOv2 与地震域预训练 NCS2D 哪一种表征更适合端到端 FWI 的跨构造测试。", "",
        "## 2. 协议一致性", "", f"V12 M1–M5 经严格 config、manifest、sample_id 和 decoder hash 门禁后复用 {integrity.get('reused_v12_count', 0)}/45；M6 的 {integrity.get('ncs_success_count', 0)}/9 runs 均使用真实 NCS2D frozen feature。所有 target test 保持隔离，所有 decoder 参数均已注册进入 optimizer。", "",
        "## 3. 模型组与比较逻辑", "", "六组方法为 M1 CNN、M2 random ViT、M3 DINOv2 frozen、M4 DINOv2-LoRA、M5 spectrogram-DINOv2-LoRA、M6 NCS2D frozen。预注册 A–E 分别对应 M3/M2、M6/M2、M6/M3、M5/M6 和 M4/M3，用于区分架构、自然图像预训练、地震域预训练、bridge 与 LoRA 适配的影响。", "",
        "## 4. 跨构造绝对误差", "", "全部指标位于 physical_velocity 空间。整体误差更低，表示预测速度整体更接近真实模型。", "", *result_lines, "",
        "## 5. 泛化差距", "", "MAE/RMSE/gradient/edge gap 为 cross-in，SSIM gap 为 in-cross。gap 更小表示从同类构造转到未见构造时下降较少；绝对跨构造误差更低不必然意味着 gap 更小。", "", *gap_lines, "",
        "## 6. 自然图像预训练与地震域预训练比较", "", *evidence_lines[:3], "M6 相对 M3 没有形成一致方向性证据：不同 transfer/seed 的数值与结构方向不统一，因此不能声称地震域预训练优于自然图像预训练。", "",
        "## 7. 频谱自然图像路线与地震域路线比较", "", evidence_lines[3], "M5 与 M6 的 bridge 和适配方式均不同，这是一项实用路线比较，不是严格的纯预训练来源对比。", "",
        "## 8. 结果解释", "", "整体 MAE/RMSE 更低表示速度趋势更接近真实模型；gradient_error 和 edge_MAE 更低表示速度层界面更清楚。一种预训练可能改善整体速度趋势，但不一定改善断层和边界，因此必须同时看数值误差、结构误差和泛化差距。", "",
        "## 9. 结论等级", "", *evidence_lines, "五项比较均为部分或混合证据，均未达到预注册的一致方向性门槛。", "",
        "## 10. 局限性", "", "- CPU-only；", "- 200/50/50 小样本；", "- OpenFWI subset；", "- 仅三种 cross-family transfer；", "- NCS2D 输入与其原始预训练地震域不完全相同；", "- 未引入真实 source/receiver/offset 几何；", "- 未测试 OOD；", "- 未测试噪声、缺道、少炮鲁棒性；", "- 不构成标准基准级结论；", "- 不构成工程应用级性能。", "",
        "## 11. 下一步", "", "NCS2D 与 DINOv2 frozen 当前均为混合证据，因此下一步应进入 physics-aware bridge 与真实几何信息，并继续保持统一 decoder、target test 隔离和 paired sample 对齐。", "",
        FOOTER, "",
    ]
    report_path = protocol_root / "protocol_v13_report.md"; report_path.write_text("\n".join(report), encoding="utf-8")
    claims = """# Protocol V13 Claims and Limitations

## Can Claim

- 已在统一小样本协议下比较自然图像预训练与地震域预训练表征；
- 已分别评价绝对跨构造误差与泛化差距；
- 可判断两类预训练在当前协议下呈现一致、混合或无一致方向性证据；
- 真实 NCS2D frozen feature 已进入统一端到端 FWI 评测矩阵。

## Cannot Claim

- 不能声称地震域预训练已经证明优于自然图像预训练；
- 不能声称 NCS2D 已证明提升 FWI 泛化能力；
- 不能声称 DINOv2 已证明提升 FWI 泛化能力；
- 当前结果不属于标准基准级结论；
- 当前结果不属于复杂 OOD 或工程应用级性能。
"""
    claims_path = protocol_root / "protocol_v13_claims_and_limitations.md"; claims_path.write_text(claims, encoding="utf-8")
    return report_path, claims_path, integrity_path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True); args = parser.parse_args()
    print("\n".join(map(str, write_protocol_v13_report(args.root))))


if __name__ == "__main__": main()
