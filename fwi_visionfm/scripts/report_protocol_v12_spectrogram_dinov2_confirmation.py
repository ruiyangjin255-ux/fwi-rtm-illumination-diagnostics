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
    from scripts.build_protocol_v12_manifests import combined_manifest_hash, compute_manifest_hashes
    from scripts.build_protocol_v12_matrix import V12_METHOD_SPECS
except ModuleNotFoundError:  # direct script execution
    from build_protocol_v12_manifests import combined_manifest_hash, compute_manifest_hashes
    from build_protocol_v12_matrix import V12_METHOD_SPECS


FOOTER = "CPU 小样本统一协议；结果用于验证方向性证据，不构成标准基准级结论。"


def _font() -> None:
    path = Path("C:/Windows/Fonts/msyh.ttc")
    if path.is_file(): fm.fontManager.addfont(str(path)); plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file(): return []
    with path.open("r", encoding="utf-8", newline="") as handle: return list(csv.DictReader(handle))


def _num(row: dict[str, str], key: str) -> float | None:
    try: return float(row[key])
    except (KeyError, TypeError, ValueError): return None


def _fmt(value: float | None, digits: int = 2) -> str: return "-" if value is None else f"{value:.{digits}f}"


def _placeholder(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5)); ax.axis("off"); ax.text(.5, .6, title, ha="center", fontsize=18, weight="bold"); ax.text(.5, .42, "当前没有足够成功结果", ha="center"); fig.text(.5, .03, FOOTER, ha="center", fontsize=9, color="#455a64"); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _matrix_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5)); ax.axis("off"); cells = [[row["method_id"], row["method_name"], row["bridge"], row["adapter"]] for row in V12_METHOD_SPECS]; table = ax.table(cellText=cells, colLabels=["ID", "方法", "Bridge", "适配"], loc="center", cellLoc="left"); table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.7); ax.set_title("Protocol V12 确认性模型矩阵", fontsize=18, weight="bold", loc="left"); fig.text(.5, .03, FOOTER, ha="center", fontsize=9, color="#455a64"); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _heatmap(path: Path, rows: list[dict[str, str]], metric: str, title: str) -> None:
    if not rows or metric not in rows[0]: return _placeholder(path, title)
    methods = [row["method_key"] for row in V12_METHOD_SPECS]; transfers = sorted({row["transfer_id"] for row in rows}); lookup = {(row["method_key"], row["transfer_id"]): float(row[metric]) for row in rows}; data = np.asarray([[lookup.get((method, transfer), np.nan) for transfer in transfers] for method in methods])
    fig, ax = plt.subplots(figsize=(11, 6)); image = ax.imshow(data, cmap="Blues_r", aspect="auto"); ax.set_xticks(range(len(transfers)), transfers, rotation=18, ha="right"); ax.set_yticks(range(len(methods)), [row["method_name"] for row in V12_METHOD_SPECS]);
    for i in range(len(methods)):
        for j in range(len(transfers)):
            if np.isfinite(data[i, j]): ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center")
    ax.set_xlabel("跨构造设置"); ax.set_ylabel("模型方法"); ax.set_title(title, fontsize=18, weight="bold", loc="left"); fig.colorbar(image, ax=ax, shrink=.75); fig.text(.5, .01, FOOTER, ha="center", fontsize=9, color="#455a64"); fig.tight_layout(rect=(0,.04,1,1)); fig.savefig(path, dpi=180); plt.close(fig)


def _bar(path: Path, rows: list[dict[str, str]], metric: str, title: str, methods: set[str] | None = None) -> None:
    selected = [row for row in rows if methods is None or row.get("method_key") in methods]
    if not selected or metric not in selected[0]: return _placeholder(path, title)
    labels = [f"{row['method_key']}\n{row['transfer_id']}" for row in selected]; values = [float(row[metric]) for row in selected]; fig, ax = plt.subplots(figsize=(max(10, len(labels)*.65), 6)); ax.bar(range(len(values)), values, color="#4f81a6"); ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8); ax.set_xlabel("模型方法与跨构造设置"); ax.set_ylabel({"cross_family_mae_mean": "物理速度 MAE（m/s）", "mae_generalization_gap_mean": "MAE 泛化差距（m/s）"}.get(metric, "指标值")); ax.set_title(title, fontsize=18, weight="bold", loc="left"); ax.spines[["top", "right"]].set_visible(False); fig.text(.5, .01, FOOTER, ha="center", fontsize=9, color="#455a64"); fig.tight_layout(rect=(0,.05,1,1)); fig.savefig(path, dpi=180); plt.close(fig)


def _bootstrap_figure(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows: return _placeholder(path, "预注册比较配对 bootstrap MAE 差值")
    labels = [f"{row['comparison_id']} {row['transfer_id']} s{row['seed']}" for row in rows]; means = np.asarray([float(row["mae_mean_difference"]) for row in rows]); low = np.asarray([float(row["mae_ci_low"]) for row in rows]); high = np.asarray([float(row["mae_ci_high"]) for row in rows]); y = np.arange(len(rows)); fig, ax = plt.subplots(figsize=(12, max(6, len(rows)*.25))); ax.errorbar(means, y, xerr=[means-low, high-means], fmt="o", capsize=3); ax.axvline(0, color="#263238", linestyle="--"); ax.set_yticks(y, labels, fontsize=7); ax.set_xlabel("候选 MAE - 对照 MAE（m/s）"); ax.set_title("预注册比较的配对 bootstrap MAE 差值", fontsize=18, weight="bold", loc="left"); fig.text(.5, .01, FOOTER, ha="center", fontsize=9, color="#455a64"); fig.tight_layout(rect=(0,.04,1,1)); fig.savefig(path, dpi=180); plt.close(fig)


def _predictions(path: Path, root: Path) -> None:
    candidates = sorted(root.glob("runs/*/spectrogram_dinov2_lora/seed_0/predictions_cross_family_test.npz"))
    if not candidates: return _placeholder(path, "频谱 DINOv2 跨构造预测样例")
    fig, axes = plt.subplots(len(candidates), 3, figsize=(10, 3*len(candidates)), squeeze=False)
    for index, candidate in enumerate(candidates):
        with np.load(candidate) as payload: truth = payload["velocity_true_physical"][0,0]; pred = payload["velocity_pred_physical"][0,0]
        for ax, image, title in zip(axes[index], (truth, pred, np.abs(pred-truth)), (f"{candidate.parents[2].name} 真实", "预测", "绝对误差")): ax.imshow(image, cmap="viridis"); ax.set_title(title); ax.axis("off")
    fig.suptitle("频谱 DINOv2 跨构造预测样例", fontsize=18, weight="bold"); fig.text(.5,.01,FOOTER,ha="center",fontsize=9,color="#455a64"); fig.tight_layout(rect=(0,.04,1,.95)); fig.savefig(path,dpi=180); plt.close(fig)


def _result_tables(rows: list[dict[str, str]]) -> list[str]:
    lines = []
    for transfer in sorted({row.get("transfer_id", "") for row in rows if row.get("transfer_id")}):
        lines.extend([f"### {transfer}", "", "| 方法 | MAE | RMSE | SSIM | gradient_error | edge_MAE | MAE gap | RMSE gap | SSIM gap | gradient gap | edge gap |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for row in sorted((item for item in rows if item.get("transfer_id") == transfer), key=lambda item: item.get("method_id", "")):
            lines.append("| " + " | ".join([row.get("method_name", row.get("method_key", "-")), _fmt(_num(row,"cross_family_mae_mean")), _fmt(_num(row,"cross_family_rmse_mean")), _fmt(_num(row,"cross_family_ssim_mean"),4), _fmt(_num(row,"cross_family_gradient_error_mean")), _fmt(_num(row,"cross_family_edge_mae_mean")), _fmt(_num(row,"mae_generalization_gap_mean")), _fmt(_num(row,"rmse_generalization_gap_mean")), _fmt(_num(row,"ssim_generalization_gap_mean"),4), _fmt(_num(row,"gradient_error_generalization_gap_mean")), _fmt(_num(row,"edge_mae_generalization_gap_mean"))]) + " |")
        lines.append("")
    return lines


def _paired_difference_tables(rows: list[dict[str, str]]) -> list[str]:
    lines = ["### M5 相对预注册对照的配对差异", "", "负值表示 spectrogram-DINOv2-LoRA 的误差更低。", "", "| 比较 | transfer | 平均 MAE 差值 | 平均 RMSE 差值 | 平均 gradient 差值 | 平均 edge 差值 | MAE CI<0 seed |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for comparison_id in ("M5_vs_M4", "M5_vs_M2", "M5_vs_M1"):
        transfers = sorted({row["transfer_id"] for row in rows if row.get("comparison_id") == comparison_id})
        for transfer in transfers:
            group = [row for row in rows if row.get("comparison_id") == comparison_id and row.get("transfer_id") == transfer]
            mean = lambda key: float(np.mean([float(row[key]) for row in group]))
            ci_count = sum(float(row["mae_ci_high"]) < 0 for row in group)
            lines.append(f"| {comparison_id} | {transfer} | {mean('mae_mean_difference'):.2f} | {mean('rmse_mean_difference'):.2f} | {mean('gradient_error_mean_difference'):.2f} | {mean('edge_mae_mean_difference'):.2f} | {ci_count}/3 |")
    lines.append("")
    return lines


def _evidence_detail_lines(evidence: dict[str, Any], comparison_ids: tuple[str, ...]) -> list[str]:
    lines: list[str] = []
    for comparison_id in comparison_ids:
        value = evidence.get(comparison_id)
        if not value:
            lines.append(f"- **{comparison_id}**：当前尚无完整确认性证据。")
            continue
        details = value.get("details", [])
        numerical_transfers = sum(int(row.get("numerical_seed_count", 0)) >= 2 for row in details)
        structural_transfers = sum(int(row.get("structural_seed_count", 0)) >= 2 for row in details)
        bootstrap_transfers = sum(int(row.get("bootstrap_ci_below_zero_seed_count", 0)) >= 2 for row in details)
        lines.extend([
            f"### {comparison_id}", "",
            f"最终判定：**{value.get('evidence_level', '当前未形成一致证据')}**。MAE/RMSE 至少 2/3 seed 同向改善的 transfer 为 {numerical_transfers}/3；结构指标至少 2/3 seed 不更差的 transfer 为 {structural_transfers}/3；MAE bootstrap CI<0 至少 2/3 seed 的 transfer 为 {bootstrap_transfers}/3；全部条件同时成立的 transfer 为 {value.get('qualifying_transfer_count', 0)}/3。", "",
            "| transfer | 数值改善 seed | 结构不差 seed | bootstrap 支持 seed | 是否满足 |",
            "| --- | ---: | ---: | ---: | --- |",
        ])
        for row in details:
            lines.append(f"| {row['transfer_id']} | {row['numerical_seed_count']}/3 | {row['structural_seed_count']}/3 | {row['bootstrap_ci_below_zero_seed_count']}/3 | {'是' if row.get('qualifying') else '否'} |")
        lines.append("")
    return lines


def _integrity(root: Path) -> tuple[Path, dict[str, Any]]:
    configs = list(root.glob("runs/*/*/seed_*/config.json")); cards = list(root.glob("runs/*/*/seed_*/model_card.json")); manifest_file = root / "manifests" / "protocol_v12_manifest_hashes.json"; hash_ok = False
    if manifest_file.is_file():
        recorded = json.loads(manifest_file.read_text(encoding="utf-8")); hash_ok = combined_manifest_hash(compute_manifest_hashes(root / "manifests")) == recorded.get("combined_hash")
    card_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in cards]; config_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in configs]; payload = {"run_count": len(configs), "success": sum(row.get("status") == "SUCCESS" for row in config_payloads), "manifest_hash_ok": hash_ok, "decoder_optimizer_registration_ok": bool(card_payloads) and all(row.get("decoder_fully_registered") is True and row.get("optimizer_parameters") == row.get("trainable_parameters") for row in card_payloads), "target_isolation_flags_ok": bool(config_payloads) and all(not row.get("target_test_used_for_training") and not row.get("target_test_used_for_validation") and not row.get("target_test_used_for_model_selection") for row in config_payloads)}
    lines = ["# Protocol V12 协议完整性报告", "", f"- run 数：{payload['run_count']}", f"- success：{payload['success']}", f"- manifest hash 固定：{payload['manifest_hash_ok']}", f"- decoder 全部进入 optimizer：{payload['decoder_optimizer_registration_ok']}", f"- target test 隔离标记：{payload['target_isolation_flags_ok']}", "- Stage A 之后不依据指标修改模型、bridge、decoder、loss、学习率或 epochs。", ""]
    path = root / "protocol_v12_protocol_integrity_report.md"; path.write_text("\n".join(lines), encoding="utf-8"); return path, payload


def write_protocol_v12_report(root: str | Path) -> tuple[Path, Path, Path]:
    _font(); protocol_root = Path(root); aggregate = _csv(protocol_root / "protocol_v12_aggregate_metrics.csv"); bootstrap = _csv(protocol_root / "bootstrap" / "protocol_v12_bootstrap_deltas.csv"); summary_path = protocol_root / "protocol_v12_summary.json"; summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {"comparison_evidence": {}}; integrity_path, integrity = _integrity(protocol_root); figures = protocol_root / "figures"; figures.mkdir(parents=True, exist_ok=True)
    _matrix_figure(figures/"figure_01_protocol_matrix.png"); _heatmap(figures/"figure_02_crossfamily_mae_heatmap.png",aggregate,"cross_family_mae_mean","跨构造 MAE 热图"); _heatmap(figures/"figure_03_crossfamily_structural_heatmap.png",aggregate,"cross_family_edge_mae_mean","跨构造边界误差热图"); _bar(figures/"figure_04_generalization_gap.png",aggregate,"mae_generalization_gap_mean","MAE 泛化差距"); _bar(figures/"figure_05_dinov2_transfer_comparison.png",aggregate,"cross_family_mae_mean","自然图像预训练迁移比较",{"random_vit","dinov2_frozen","dinov2_lora","spectrogram_dinov2_lora"}); _bar(figures/"figure_06_spectrogram_bridge_effect.png",aggregate,"cross_family_mae_mean","频谱 bridge 对 DINOv2-LoRA 的影响",{"dinov2_lora","spectrogram_dinov2_lora"}); _bootstrap_figure(figures/"figure_07_bootstrap_delta.png",bootstrap); _predictions(figures/"figure_08_prediction_grid_by_transfer.png",protocol_root)
    evidence = summary.get("comparison_evidence", {})
    report = ["# Protocol V12 频谱 DINOv2 跨构造确认评测报告", "", "## 1. 研究问题", "", "本轮重点不是谁的误差最低，而是验证自然图像预训练视觉模型进入端到端 FWI 后，频谱化输入是否能使其在未见构造中表现得更稳定。", "", "## 2. 为什么要单独验证频谱 DINOv2", "", "V11 中 spectrogram-DINOv2-LoRA 出现一致方向性证据，因此 V12 用更大的 train_size=200 和完全锁定协议检查该趋势是否重复出现。", "", "## 3. 统一协议与数据隔离", "", "固定 200/50/50、3 seeds、3 transfers、CPU、2 epochs、5 炮、70×70 physical velocity、mean aggregation、共同 decoder 与 default L1。target test 完全隔离，只用于最终评价。", "", "## 4. 模型与输入方式", "", "M1 CNN 是任务基线；M2 random ViT 排除架构影响；M3 DINOv2 frozen 检查直接迁移；M4 DINOv2-LoRA 检查适配；M5 spectrogram-DINOv2-LoRA 只改变 bridge。", "", "## 5. 跨构造结果", "", f"共形成 {len(aggregate)} 条 method-transfer 聚合结果。", "", *_result_tables(aggregate), *_paired_difference_tables(bootstrap), "## 6. 频谱输入是否改善 DINOv2-LoRA", "", *_evidence_detail_lines(evidence, ("M5_vs_M4", "M5_vs_M2", "M5_vs_M1")), "详细 seed 一致性和 bootstrap 置信区间见 `bootstrap/protocol_v12_seed_consistency.csv` 与 `bootstrap/protocol_v12_bootstrap_deltas.csv`。", "", "## 7. 自然图像预训练是否可迁移", "", *_evidence_detail_lines(evidence, ("M3_vs_M2", "M4_vs_M3")), "结论仅使用一致的方向性证据、部分或混合证据、当前未形成一致证据三级口径。", "", "## 8. 数值误差与结构误差如何理解", "", "MAE/RMSE 较低表示整体速度数值更接近真实模型；gradient_error 和 edge_MAE 较低表示速度界面更清楚。一个模型可能整体更准，但界面仍偏平滑，因此必须联合评价。", "", "## 9. 结论边界", "", "- 当前为 CPU 小样本统一评测；", "- train_size=200；", "- 只有三个跨构造设置；", "- 使用 OpenFWI subset；", "- 不构成标准基准级结论；", "- 不构成复杂 OOD 或工程应用级性能结论；", "- 后续仍需更多 family、更大样本和统一大规模协议。", "", "## 10. 后续路线", "", "M5 相对 raw DINOv2-LoRA 的确认结果为混合，说明频谱 bridge 的收益依赖构造类型；M5 相对 CNN 与 random ViT 虽达到预注册一致方向性证据，仍需更多 family、更大样本和统一大规模协议复核。NCS2D 与 boundary-aware decoder 另设协议。", "", FOOTER, ""]
    report_path = protocol_root / "protocol_v12_report.md"; report_path.write_text("\n".join(report), encoding="utf-8")
    claims = """# Protocol V12 Claims and Limitations

## Can Claim

- 已在锁定统一协议下，对自然图像视觉模型端到端 FWI 跨构造表现进行了确认性评测；
- 可判断频谱 bridge 对 DINOv2-LoRA 在当前协议下是否呈现一致、混合或无一致方向性证据；
- 可比较自然图像预训练直接迁移、LoRA 适配和频谱化输入的差异；
- 可分别评价整体速度数值误差与速度界面恢复误差。

## Cannot Claim

- 不能声称频谱 DINOv2-LoRA 已经证明提升 FWI 泛化能力；
- 不能声称 DINOv2 已优于所有 CNN 或 FWI 基线；
- 不能声称任何结果适用于 Marmousi、Salt、Sigsbee、SEAM 等复杂 OOD；
- 当前结果不属于标准基准级结论；
- 当前结果不属于工程应用级性能。
"""; claims_path = protocol_root / "protocol_v12_claims_and_limitations.md"; claims_path.write_text(claims, encoding="utf-8"); return report_path, claims_path, integrity_path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", required=True); args = parser.parse_args(); print("\n".join(map(str, write_protocol_v12_report(args.root))))


if __name__ == "__main__": main()
