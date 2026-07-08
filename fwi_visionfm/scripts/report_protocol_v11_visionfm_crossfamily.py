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
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

from fwi_visionfm.models.protocol_v11_model_registry import METHOD_SPECS


FOOTER = "CPU 小样本统一协议；不构成标准基准级结论。"


def _setup_font() -> None:
    path = Path("C:/Windows/Fonts/msyh.ttc")
    if path.is_file():
        fm.fontManager.addfont(str(path))
        plt.rcParams["font.family"] = "Microsoft YaHei"
    plt.rcParams["axes.unicode_minus"] = False


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file(): return []
    with path.open("r", encoding="utf-8", newline="") as handle: return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _transfer_result_tables(rows: list[dict[str, str]]) -> list[str]:
    lines: list[str] = []
    for transfer in sorted({row.get("transfer_id", "") for row in rows if row.get("transfer_id")}):
        group = [row for row in rows if row.get("transfer_id") == transfer]
        lookup = {row.get("method_key"): row for row in group}
        cnn_mae = _number(lookup.get("cnn_baseline", {}), "cross_family_mae_mean")
        random_mae = _number(lookup.get("random_vit", {}), "cross_family_mae_mean")
        lines.extend([
            f"### {transfer}", "",
            "| 方法 | MAE | RMSE | SSIM | gradient_error | edge_MAE | MAE gap | RMSE gap | structural gap | ΔMAE vs CNN | ΔMAE vs random ViT |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for row in sorted(group, key=lambda item: item.get("method_id", item.get("method_key", ""))):
            mae = _number(row, "cross_family_mae_mean")
            delta_cnn = None if mae is None or cnn_mae is None else mae - cnn_mae
            delta_random = None if mae is None or random_mae is None else mae - random_mae
            lines.append("| " + " | ".join([
                row.get("method_name", row.get("method_key", "-")), _fmt(mae),
                _fmt(_number(row, "cross_family_rmse_mean")), _fmt(_number(row, "cross_family_ssim_mean"), 4),
                _fmt(_number(row, "cross_family_gradient_error_mean")), _fmt(_number(row, "cross_family_edge_mae_mean")),
                _fmt(_number(row, "mae_generalization_gap_mean")), _fmt(_number(row, "rmse_generalization_gap_mean")),
                _fmt(_number(row, "edge_mae_generalization_gap_mean")), _fmt(delta_cnn), _fmt(delta_random),
            ]) + " |")
        lines.append("")
    return lines


def _pair_win_count(rows: list[dict[str, str]], candidate: str, reference: str, metric: str) -> tuple[int, int]:
    lookup = {(row.get("transfer_id"), row.get("method_key")): row for row in rows}
    transfers = sorted({key[0] for key in lookup if key[1] == candidate and (key[0], reference) in lookup})
    wins = 0
    for transfer in transfers:
        candidate_value = _number(lookup[(transfer, candidate)], metric)
        reference_value = _number(lookup[(transfer, reference)], metric)
        if candidate_value is not None and reference_value is not None:
            wins += int(candidate_value > reference_value if metric.endswith("ssim_mean") else candidate_value < reference_value)
    return wins, len(transfers)


def _comparison_summary(rows: list[dict[str, str]], candidate: str, reference: str) -> str:
    mae_wins, total = _pair_win_count(rows, candidate, reference, "cross_family_mae_mean")
    rmse_wins, _ = _pair_win_count(rows, candidate, reference, "cross_family_rmse_mean")
    grad_wins, _ = _pair_win_count(rows, candidate, reference, "cross_family_gradient_error_mean")
    edge_wins, _ = _pair_win_count(rows, candidate, reference, "cross_family_edge_mae_mean")
    if total == 0:
        return "暂无可配对的聚合结果。"
    return f"在 {total} 个 transfer 的跨构造均值比较中，MAE 胜出 {mae_wins} 个、RMSE 胜出 {rmse_wins} 个、gradient_error 胜出 {grad_wins} 个、edge_MAE 胜出 {edge_wins} 个。"


def _placeholder(path: Path, title: str, message: str = "当前没有足够的成功结果可绘图") -> None:
    fig, ax = plt.subplots(figsize=(10, 5)); ax.axis("off"); ax.text(0.5, 0.62, title, ha="center", fontsize=18, weight="bold"); ax.text(0.5, 0.43, message, ha="center", fontsize=12); ax.text(0.5, 0.08, FOOTER, ha="center", fontsize=10, color="#455a64"); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _figure_model_matrix(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 6)); ax.axis("off")
    columns = ["ID", "方法", "预训练来源", "Bridge"]
    cells = [[s["method_id"], s["method_name"], s["pretraining_source"], s["bridge"]] for s in METHOD_SPECS]
    table = ax.table(cellText=cells, colLabels=columns, loc="center", cellLoc="left", colLoc="left", colWidths=[0.08, 0.28, 0.28, 0.32]); table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.7)
    ax.set_title("Protocol V11 六组统一模型矩阵", fontsize=18, weight="bold", loc="left", pad=18); ax.text(0.5, 0.02, FOOTER, transform=ax.transAxes, ha="center", fontsize=10, color="#455a64"); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _heatmap(path: Path, rows: list[dict[str, str]], metric: str, title: str) -> None:
    if not rows or metric not in rows[0]: return _placeholder(path, title)
    methods = [s["method_key"] for s in METHOD_SPECS]; transfers = sorted({r["transfer_id"] for r in rows})
    lookup = {(r["method_key"], r["transfer_id"]): float(r[metric]) for r in rows}
    matrix = np.array([[lookup.get((m, t), np.nan) for t in transfers] for m in methods])
    fig, ax = plt.subplots(figsize=(11, 6)); image = ax.imshow(matrix, cmap="Blues_r", aspect="auto")
    ax.set_xticks(range(len(transfers)), transfers, rotation=18, ha="right"); ax.set_yticks(range(len(methods)), [s["method_name"] for s in METHOD_SPECS])
    for i in range(len(methods)):
        for j in range(len(transfers)):
            if np.isfinite(matrix[i, j]): ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(title, fontsize=18, weight="bold", loc="left"); fig.colorbar(image, ax=ax, shrink=0.75); fig.text(0.5, 0.01, FOOTER, ha="center", fontsize=10, color="#455a64"); fig.tight_layout(rect=(0,0.04,1,1)); fig.savefig(path, dpi=180); plt.close(fig)


def _bar_metric(path: Path, rows: list[dict[str, str]], metric: str, title: str, methods: list[str] | None = None) -> None:
    if not rows or metric not in rows[0] or "method_key" not in rows[0]: return _placeholder(path, title)
    selected = [r for r in rows if methods is None or r.get("method_key") in methods]
    if not selected: return _placeholder(path, title)
    labels = [f"{r['method_key']}\n{r['transfer_id']}" for r in selected]; values = [float(r[metric]) for r in selected]
    fig, ax = plt.subplots(figsize=(max(10, len(labels)*0.6), 6)); ax.bar(range(len(values)), values, color="#4f81a6", edgecolor="#263238"); ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8); ax.set_title(title, fontsize=18, weight="bold", loc="left"); ax.grid(axis="x", visible=False); ax.spines[["top","right"]].set_visible(False); fig.text(0.5, 0.01, FOOTER, ha="center", fontsize=10, color="#455a64"); fig.tight_layout(rect=(0,0.05,1,1)); fig.savefig(path, dpi=180); plt.close(fig)


def _bootstrap_figure(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows: return _placeholder(path, "相对基线的配对 bootstrap MAE 差值")
    labels = [f"{r['method_key']} vs {r['baseline_method']}\n{r['transfer_id']} s{r['seed']}" for r in rows]
    means = np.array([float(r["mean_difference"]) for r in rows]); low = np.array([float(r["ci_low"]) for r in rows]); high = np.array([float(r["ci_high"]) for r in rows])
    y = np.arange(len(rows)); fig, ax = plt.subplots(figsize=(12, max(6, len(rows)*0.25))); ax.errorbar(means, y, xerr=[means-low, high-means], fmt="o", color="#1565c0", ecolor="#78909c", capsize=3); ax.axvline(0, color="#263238", linestyle="--"); ax.set_yticks(y, labels, fontsize=7); ax.set_xlabel("候选 MAE - 基线 MAE（m/s，负值有利于候选）"); ax.set_title("相对 CNN 与 random ViT 的配对 bootstrap 差值", fontsize=18, weight="bold", loc="left"); fig.text(0.5, 0.01, FOOTER, ha="center", fontsize=10, color="#455a64"); fig.tight_layout(rect=(0,0.04,1,1)); fig.savefig(path, dpi=180); plt.close(fig)


def _prediction_grid(path: Path, root: Path) -> None:
    candidates = sorted(root.glob("runs/*/*/seed_0/predictions_cross_family_test.npz"))
    if not candidates: return _placeholder(path, "按 transfer 展示跨类型预测")
    selected = []
    seen = set()
    for candidate in candidates:
        transfer = candidate.parents[2].name
        if transfer not in seen: selected.append((transfer, candidate)); seen.add(transfer)
    fig, axes = plt.subplots(len(selected), 3, figsize=(10, 3*len(selected)), squeeze=False)
    for i, (transfer, candidate) in enumerate(selected):
        with np.load(candidate) as payload: truth=payload["velocity_true_physical"][0,0]; pred=payload["velocity_pred_physical"][0,0]
        for ax,image,title in zip(axes[i],[truth,pred,np.abs(pred-truth)],[f"{transfer} 真实","预测","绝对误差"]): ax.imshow(image,cmap="viridis"); ax.set_title(title); ax.axis("off")
    fig.suptitle("跨类型预测样例",fontsize=18,weight="bold"); fig.text(0.5,0.01,FOOTER,ha="center",fontsize=10,color="#455a64"); fig.tight_layout(rect=(0,0.04,1,0.95)); fig.savefig(path,dpi=180); plt.close(fig)


def _generate_figures(root: Path, aggregate: list[dict[str, str]], bootstrap: list[dict[str, str]]) -> None:
    _setup_font()
    out=root/"figures"; out.mkdir(parents=True,exist_ok=True)
    _figure_model_matrix(out/"figure_01_model_matrix.png")
    _heatmap(out/"figure_02_crossfamily_mae_heatmap.png",aggregate,"cross_family_mae_mean","跨类型 MAE 热图")
    _heatmap(out/"figure_03_crossfamily_structural_heatmap.png",aggregate,"cross_family_edge_mae_mean","跨类型边界平均绝对误差热图")
    _bar_metric(out/"figure_04_generalization_gap.png",aggregate,"mae_generalization_gap_mean","MAE 同类型到跨类型差距")
    _bar_metric(out/"figure_05_dinov2_transfer_comparison.png",aggregate,"cross_family_mae_mean","自然图像 DINOv2 迁移路线比较",["random_vit","dinov2_frozen","dinov2_lora","spectrogram_dinov2_lora"])
    _bar_metric(out/"figure_06_natural_vs_seismic_pretraining.png",aggregate,"cross_family_mae_mean","自然图像预训练与地震域预训练方向性比较",["dinov2_lora","spectrogram_dinov2_lora","ncs2d_frozen"])
    _bootstrap_figure(out/"figure_07_bootstrap_delta_vs_baselines.png",bootstrap)
    _prediction_grid(out/"figure_08_prediction_grid_by_transfer.png",root)


def write_protocol_v11_report(*, root: str | Path) -> tuple[Path, Path]:
    protocol_root=Path(root); aggregate=_csv(protocol_root/"protocol_v11_aggregate_metrics.csv"); bootstrap=_csv(protocol_root/"bootstrap"/"protocol_v11_bootstrap_deltas.csv")
    _generate_figures(protocol_root,aggregate,bootstrap)
    availability={}
    av_path=protocol_root/"availability"/"protocol_v11_availability.json"
    if av_path.is_file(): availability=json.loads(av_path.read_text(encoding="utf-8"))
    levels={r.get("method_name",r.get("method_id","method")):r.get("evidence_level","当前未形成一致证据") for r in aggregate}
    available_transfers=len({r.get("transfer_id") for r in aggregate if r.get("transfer_id")})
    if not levels:
        levels={"DINOv2 frozen":"当前未形成一致证据"}
    model_status=[f"- {row.get('method_key')}：{row.get('status')}。" for row in availability.get("backbones",[])] or ["- 可用性结果尚未生成。"]
    result_tables = _transfer_result_tables(aggregate)
    report=[
        "# Protocol V11 视觉模型跨构造评测报告","","## 1. 研究问题","","本轮不是单纯比较谁的误差最低，而是回答“视觉模型是否适用于端到端 FWI”，并检查其在未见构造中的表现是否比任务专用基线更稳定。","",
        "## 2. 统一实验协议","","FlatVel_A、CurveVel_A 与 FlatFault_A 组成三种跨类型设置；固定 100/50/50 样本、2 epochs、seed=0/1/2、5 炮、70×70 输出、CPU、mean 聚合、共同 bounded decoder 与 default L1。目标域测试集只用于最终评价，不参与训练、模型选择、提前停止或调参。","",
        "## 3. 模型可用性","",*model_status,"",
        "## 4. 跨构造结果","",f"已形成 {len(aggregate)} 条 method-transfer 聚合结果，覆盖 {available_transfers} 个可运行跨类型设置。表中为三个 seed 的均值，差值为候选减基线；负的 MAE 差值有利于候选。完整数值与标准差见 `protocol_v11_aggregate_metrics.csv`。" if aggregate else "当前尚无成功聚合结果。","",*result_tables,
        "## 5. 自然图像预训练是否可迁移","","比较 random ViT、DINOv2 frozen、DINOv2-LoRA 与 spectrogram-DINOv2-LoRA。结论仅按预注册规则判为一致、混合或无一致证据。","",
        f"- DINOv2 frozen 的直接迁移：{_comparison_summary(aggregate, 'dinov2_frozen', 'random_vit')}",
        f"- LoRA 相对 frozen DINOv2：{_comparison_summary(aggregate, 'dinov2_lora', 'dinov2_frozen')}",
        f"- spectrogram bridge 相对 raw-envelope DINOv2-LoRA：{_comparison_summary(aggregate, 'spectrogram_dinov2_lora', 'dinov2_lora')}","",
        "## 6. 自然图像预训练与地震域预训练对比","","DINOv2-LoRA、spectrogram-DINOv2-LoRA 与 NCS2D frozen 的比较用于判断预训练来源在当前小样本协议下的方向性差异，不等价于统一大规模 benchmark。V9 的 NCS2D + boundary-aware decoder 仅作为补充参考，不进入主矩阵排名。","",
        f"- NCS2D frozen 相对 DINOv2-LoRA：{_comparison_summary(aggregate, 'ncs2d_frozen', 'dinov2_lora')}",
        f"- NCS2D frozen 相对 spectrogram-DINOv2-LoRA：{_comparison_summary(aggregate, 'ncs2d_frozen', 'spectrogram_dinov2_lora')}","",
        "## 7. 结构恢复与整体数值误差","","MAE/RMSE 较低表示整体速度更接近真实模型；gradient_error/edge_MAE 较低表示速度界面和边界更清晰。一个模型可能整体误差较小，但速度界面仍不够清楚，因此不能只看单一数值指标。","",
        "## 8. 结论判读","",*[f"- {name}：{level}。" for name,level in sorted(levels.items())],"",
        "## 9. 局限性","","- CPU-only；","- 100/50/50 小样本；","- OpenFWI subset；","- 当前是统一小样本协议，不构成标准基准级结论；","- 不能直接外推到 Marmousi、Salt、Sigsbee、SEAM 等复杂 OOD；","- NCS2D 输入与其原始地震预训练域并不完全相同；","- 未引入真实 source/receiver/offset 几何信息；","- V11 主矩阵不包含 boundary auxiliary，结构增强结果另行解释。","",
        "## 10. 下一步","","若 DINOv2-LoRA 或频谱 DINOv2 路线出现一致方向性证据，将进入更大样本或更多 seed 验证；若 NCS2D 更稳定，将继续独立验证 NCS2D + boundary-aware decoder；若自然图像预训练未形成一致证据，则将域差异作为重要负结果；若 bridge 影响较大，V12 转向 physics-aware bridge 与真实几何输入。",
    ]
    if available_transfers<2: report.extend(["","**协议未完整覆盖，不形成跨构造结论。**"])
    report_path=protocol_root/"protocol_v11_report.md"; report_path.write_text("\n".join(report)+"\n",encoding="utf-8")
    claims="""# Protocol V11 Claims and Limitations

## Can Claim

- 已建立统一的视觉模型端到端 FWI 跨构造评测协议；
- 在固定小样本、固定 bridge、固定 decoder 和固定指标下比较多类表征来源；
- 可判断自然图像预训练、地震域预训练及频谱 bridge 在当前协议下是否呈现一致、混合或无一致证据；
- 可重复评估整体数值误差与结构恢复误差。

## Cannot Claim

- 不能声称任一模型已经证明提升 FWI 泛化能力；
- 不能声称任一模型已经在复杂 OOD 基准上有效；
- 不能声称任一模型已经优于所有 CNN/FWI 基线；
- 当前结果不属于 benchmark-level proof；
- 当前结果不属于实际工程应用级性能。
"""
    claims_path=protocol_root/"protocol_v11_claims_and_limitations.md"; claims_path.write_text(claims,encoding="utf-8")
    return report_path,claims_path


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--root",required=True); args=parser.parse_args(); paths=write_protocol_v11_report(root=args.root); print("\n".join(map(str,paths)))


if __name__=="__main__": main()
