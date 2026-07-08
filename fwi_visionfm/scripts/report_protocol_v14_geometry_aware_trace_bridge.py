# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _count_prefix(rows: list[dict[str, Any]], prefix: str) -> int:
    return sum(str(row.get("comparison_id", "")).startswith(prefix) for row in rows)


def _load_matplotlib():
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _figure_footer(fig: Any) -> None:
    fig.text(0.5, 0.02, "CPU 小样本统一协议；结果用于检验方向性证据，不构成标准基准级结论。", ha="center", fontsize=9)


def _write_placeholder_figure(path: Path, *, title: str, message: str) -> None:
    plt = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    ax.text(0.5, 0.65, title, ha="center", va="center", fontsize=16, weight="bold")
    ax.text(0.5, 0.40, message, ha="center", va="center", fontsize=11, wrap=True)
    _figure_footer(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "nan"):
        return None
    return float(value)


def _group_mean(rows: list[dict[str, Any]], group_keys: tuple[str, ...], value_key: str) -> dict[tuple[str, ...], float]:
    grouped: dict[tuple[str, ...], list[float]] = {}
    for row in rows:
        value = _safe_float(row.get(value_key))
        if value is None:
            continue
        key = tuple(str(row[k]) for k in group_keys)
        grouped.setdefault(key, []).append(value)
    return {key: sum(values) / len(values) for key, values in grouped.items()}


def _plot_geometry_provenance(path: Path, geometry: dict[str, Any]) -> None:
    plt = _load_matplotlib()
    rows = geometry.get("rows", [])
    labels = [str(row.get("field_name", "")) for row in rows] or ["geometry"]
    values = [1 if row.get("available") else 0 for row in rows] or [0]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(range(len(labels)), values, color=["#2a9d8f" if v else "#cfd8dc" for v in values])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["无", "有"])
    ax.set_title(f"几何元数据审计：{geometry.get('geometry_provenance', 'UNAVAILABLE')}")
    ax.set_ylabel("可用性")
    _figure_footer(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_bridge_design(path: Path) -> None:
    plt = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axis("off")
    boxes = [
        (0.08, 0.55, 0.18, 0.18, "B0\n原始多通道波形"),
        (0.31, 0.55, 0.18, 0.18, "B1\n几何编码"),
        (0.54, 0.55, 0.18, 0.18, "B2\n邻道+整炮上下文"),
        (0.77, 0.55, 0.18, 0.18, "B3\n多尺度+频带条件"),
    ]
    for x, y, w, h, text in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor="#dceefb", edgecolor="#2f6b8a", linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)
    for idx in range(len(boxes) - 1):
        x = boxes[idx][0] + boxes[idx][2]
        ax.annotate("", xy=(boxes[idx + 1][0], 0.64), xytext=(x, 0.64), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(0.5, 0.22, "固定 backbone + 固定 decoder + 固定 loss + 固定训练预算", ha="center", fontsize=12)
    ax.set_title("几何感知 Trace Bridge 逐层设计", fontsize=15)
    _figure_footer(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_method_bridge_bars(path: Path, aggregate: list[dict[str, Any]], value_key: str, title: str, ylabel: str) -> None:
    plt = _load_matplotlib()
    methods = ["M3", "M6"]
    bridges = ["B0", "B1", "B2", "B3"]
    values = _group_mean(aggregate, ("method_id", "bridge_id"), value_key)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    width = 0.36
    x = list(range(len(bridges)))
    for idx, method in enumerate(methods):
        series = [values.get((method, bridge), float("nan")) for bridge in bridges]
        offset = [item + (idx - 0.5) * width for item in x]
        ax.bar(offset, series, width=width, label=method)
    ax.set_xticks(x)
    ax.set_xticklabels(bridges)
    ax.set_title(title)
    ax.set_xlabel("Bridge")
    ax.set_ylabel(ylabel)
    ax.legend(title="方法")
    _figure_footer(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_gain_bars(path: Path, gain: list[dict[str, Any]], title: str, filter_ids: list[str] | None = None) -> None:
    rows = [row for row in gain if filter_ids is None or row.get("comparison_id") in filter_ids]
    values = [_safe_float(row.get("delta_mae")) for row in rows]
    if not rows or all(value is None for value in values):
        _write_placeholder_figure(path, title=title, message="当前没有可用于绘制的增益数值。")
        return
    plt = _load_matplotlib()
    labels = [str(row["comparison_id"]) for row in rows]
    vals = [0.0 if value is None else value for value in values]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(range(len(labels)), vals, color=["#2a9d8f" if value < 0 else "#e76f51" for value in vals])
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("delta MAE")
    ax.set_title(title)
    _figure_footer(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_generalization_gap(path: Path, per_run: list[dict[str, Any]]) -> None:
    grouped = {}
    for row in per_run:
        in_family = _safe_float(row.get("in_family_MAE"))
        cross = _safe_float(row.get("cross_family_MAE"))
        if in_family is None or cross is None:
            continue
        key = (str(row["method_id"]), str(row["bridge_id"]))
        grouped.setdefault(key, []).append(cross - in_family)
    if not grouped:
        _write_placeholder_figure(path, title="跨构造泛化差距", message="当前没有可计算的 in-family 与 cross-family 配对指标。")
        return
    plt = _load_matplotlib()
    labels = [f"{method}-{bridge}" for method, bridge in grouped]
    vals = [sum(values) / len(values) for values in grouped.values()]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.bar(range(len(labels)), vals, color="#457b9d")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("cross MAE - in-family MAE")
    ax.set_title("各 bridge 的跨构造泛化差距")
    _figure_footer(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_bootstrap(path: Path, bootstrap_rows: list[dict[str, Any]]) -> None:
    if not bootstrap_rows:
        _write_placeholder_figure(path, title="Trace Bridge Bootstrap 效应", message="当前没有 bootstrap 结果。")
        return
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in bootstrap_rows:
        grouped.setdefault(str(row["comparison_id"]), []).append(row)
    labels = list(grouped.keys())
    means = [sum(float(item["delta_mae_mean"]) for item in rows) / len(rows) for rows in grouped.values()]
    lows = [min(float(item["mae_ci_low"]) for item in rows) for rows in grouped.values()]
    highs = [max(float(item["mae_ci_high"]) for item in rows) for rows in grouped.values()]
    plt = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = list(range(len(labels)))
    ax.bar(x, means, color=["#2a9d8f" if value < 0 else "#e76f51" for value in means])
    ax.errorbar(x, means, yerr=[[mean - low for mean, low in zip(means, lows)], [high - mean for mean, high in zip(means, highs)]], fmt="none", ecolor="black", capsize=4)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("MAE delta")
    ax.set_title("Trace Bridge 配对 Bootstrap 效应")
    _figure_footer(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_robustness(path: Path, robustness_rows: list[dict[str, Any]]) -> None:
    usable = [row for row in robustness_rows if row.get("perturbation") != "clean" and _safe_float(row.get("degradation")) is not None]
    if not usable:
        clean_rows = [row for row in robustness_rows if row.get("perturbation") == "clean" and row.get("metric_name") in {"mae", "gradient_error"}]
        status_rows = [row for row in robustness_rows if row.get("perturbation") != "clean"]
        plt = _load_matplotlib()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        ax0, ax1 = axes
        grouped = {}
        for row in clean_rows:
            key = (str(row["method_id"]), str(row["bridge_id"]))
            grouped.setdefault(key, {})[str(row["metric_name"])] = _safe_float(row.get("metric_value")) or 0.0
        labels = [f"{method}-{bridge}" for method, bridge in grouped] or ["无 clean 基线"]
        mae_vals = [grouped[key].get("mae", 0.0) for key in grouped] or [0.0]
        grad_vals = [grouped[key].get("gradient_error", 0.0) for key in grouped] or [0.0]
        x = list(range(len(labels)))
        width = 0.36
        ax0.bar([item - width / 2 for item in x], mae_vals, width=width, label="MAE")
        ax0.bar([item + width / 2 for item in x], grad_vals, width=width, label="gradient_error")
        ax0.set_xticks(x)
        ax0.set_xticklabels(labels, rotation=25, ha="right")
        ax0.set_title("已落地 clean 基线")
        ax0.set_ylabel("指标值")
        ax0.legend()

        perturbations = ["few_shot_3", "missing_receivers_30", "noise_snr10"]
        status_keys = sorted({(str(row["method_id"]), str(row["bridge_id"])) for row in status_rows})
        if not status_keys:
            status_keys = [("M3", "B0"), ("M3", "B3"), ("M6", "B0"), ("M6", "B3")]
        matrix = []
        for key in status_keys:
            row_values = []
            for perturbation in perturbations:
                matched = [row for row in status_rows if (str(row["method_id"]), str(row["bridge_id"])) == key and str(row["perturbation"]) == perturbation]
                available = any(str(row.get("status")) == "AVAILABLE" for row in matched)
                row_values.append(1.0 if available else 0.0)
            matrix.append(row_values)
        im = ax1.imshow(matrix, cmap="Greys", vmin=0.0, vmax=1.0, aspect="auto")
        ax1.set_xticks(range(len(perturbations)))
        ax1.set_xticklabels(perturbations, rotation=20, ha="right")
        ax1.set_yticks(range(len(status_keys)))
        ax1.set_yticklabels([f"{method}-{bridge}" for method, bridge in status_keys])
        ax1.set_title("扰动重评可用性")
        for r, row_values in enumerate(matrix):
            for c, value in enumerate(row_values):
                ax1.text(c, r, "可用" if value > 0.5 else "缺 checkpoint", ha="center", va="center", fontsize=9, color="black")
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
        fig.suptitle("扰动退化对比", fontsize=15)
        _figure_footer(fig)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return
    plt = _load_matplotlib()
    labels = [f"{row['method_id']}-{row['bridge_id']}-{row['perturbation']}" for row in usable[:16]]
    values = [float(row["degradation"]) for row in usable[:16]]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(range(len(labels)), values, color="#8d99ae")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("degradation")
    ax.set_title("少炮/缺道/噪声退化比较")
    _figure_footer(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_prediction_tile(path: Path, protocol_root: Path) -> None:
    plt = _load_matplotlib()
    import numpy as np

    npz_paths = []
    for candidate in [
        protocol_root / "runs" / "flatvel_a_to_curvevel_a" / "dinov2_frozen" / "seed_0" / "B0" / "predictions_cross_family_test.npz",
        protocol_root / "runs" / "flatvel_a_to_curvevel_a" / "dinov2_frozen" / "seed_0" / "B3" / "predictions_cross_family_test.npz",
        protocol_root / "runs" / "flatvel_a_to_curvevel_a" / "ncs2d_frozen" / "seed_0" / "B0" / "predictions_cross_family_test.npz",
        protocol_root / "runs" / "flatvel_a_to_curvevel_a" / "ncs2d_frozen" / "seed_0" / "B3" / "predictions_cross_family_test.npz",
    ]:
        if candidate.exists():
            npz_paths.append(candidate)
    if not npz_paths:
        _write_placeholder_figure(path, title="Bridge 与鲁棒性预测示例", message="当前没有可拼接的预测图。")
        return
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    axes = axes.ravel()
    for ax, npz_path in zip(axes, npz_paths):
        with np.load(npz_path, allow_pickle=True) as payload:
            pred = np.asarray(payload["velocity_pred_physical"], dtype=np.float32)
            target = np.asarray(payload["velocity_true_physical"], dtype=np.float32)
        if pred.ndim == 4 and pred.shape[1] == 1:
            pred = pred[:, 0]
        if target.ndim == 4 and target.shape[1] == 1:
            target = target[:, 0]
        errors = np.mean(np.abs(pred - target), axis=(1, 2))
        index = int(np.argsort(errors)[len(errors) // 2])
        panel = np.concatenate([target[index], pred[index], np.abs(pred[index] - target[index])], axis=1)
        ax.imshow(panel, aspect="auto", cmap="viridis")
        ax.set_title(npz_path.parent.parent.name + " / " + npz_path.parent.name)
        ax.axis("off")
    for ax in axes[len(npz_paths):]:
        ax.axis("off")
    fig.suptitle("Bridge 预测示例与当前鲁棒性上下文", fontsize=14)
    _figure_footer(fig)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _generate_figures(protocol_root: Path, *, geometry: dict[str, Any], aggregate: list[dict[str, Any]], per_run: list[dict[str, Any]], gain: list[dict[str, Any]], bootstrap_rows: list[dict[str, Any]], robustness_rows: list[dict[str, Any]]) -> None:
    figures = protocol_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _plot_geometry_provenance(figures / "figure_01_geometry_provenance.png", geometry)
    _plot_bridge_design(figures / "figure_02_geometry_aware_trace_bridge_design.png")
    _plot_method_bridge_bars(figures / "figure_03_crossfamily_mae_by_bridge.png", aggregate, "cross_family_MAE", "跨构造 MAE 按 Bridge 对比", "跨构造 MAE")
    _plot_method_bridge_bars(figures / "figure_04_crossfamily_structural_error_by_bridge.png", aggregate, "cross_family_gradient_error", "跨构造结构误差按 Bridge 对比", "gradient_error")
    _plot_gain_bars(figures / "figure_05_incremental_bridge_gain.png", gain, "逐层 Bridge 增益（delta MAE）")
    _plot_gain_bars(figures / "figure_06_ncs_vs_dinov2_geometry_gain.png", gain, "NCS2D 与 DINOv2 的几何增益比较", ["M3_B3_vs_B0", "M6_B3_vs_B0", "M6_B3_vs_M3_B3"])
    _plot_generalization_gap(figures / "figure_07_generalization_gap_by_bridge.png", per_run)
    _plot_bootstrap(figures / "figure_08_bootstrap_trace_bridge_effect.png", bootstrap_rows)
    _plot_robustness(figures / "figure_09_robustness_degradation.png", robustness_rows)
    _plot_prediction_tile(figures / "figure_10_prediction_grid_bridge_and_robustness.png", protocol_root)


def report_protocol_v14_geometry_aware_trace_bridge(*, root: str | Path) -> dict[str, Any]:
    protocol_root = Path(root)
    aggregate = _read_csv(protocol_root / "protocol_v14_aggregate_metrics.csv")
    gain = _read_csv(protocol_root / "protocol_v14_geometry_gain.csv")
    per_run = _read_csv(protocol_root / "protocol_v14_per_run_metrics.csv")
    unsuccessful = _read_csv(protocol_root / "protocol_v14_unsuccessful_runs.csv")
    incomplete = _read_csv(protocol_root / "protocol_v14_incomplete_outputs.csv")
    bootstrap_rows = _read_csv(protocol_root / "bootstrap" / "protocol_v14_bootstrap_deltas.csv")
    robustness_rows = _read_csv(protocol_root / "robustness" / "protocol_v14_robustness_metrics.csv")
    summary = _read_json(protocol_root / "protocol_v14_summary.json")
    audit_path = protocol_root / "geometry_audit.json"
    if not audit_path.exists():
        audit_path = protocol_root / "geometry_audit" / "geometry_audit.json"
    geometry = _read_json(audit_path) if audit_path.exists() else {"geometry_provenance": "UNAVAILABLE"}
    _generate_figures(protocol_root, geometry=geometry, aggregate=aggregate, per_run=per_run, gain=gain, bootstrap_rows=bootstrap_rows, robustness_rows=robustness_rows)

    incomplete_examples = []
    for row in incomplete[:6]:
        incomplete_examples.append(
            f"  - {row.get('run_id', '')}: 缺 {row.get('missing_required_files', '')}"
        )
    unsuccessful_examples = []
    for row in unsuccessful[:6]:
        unsuccessful_examples.append(
            f"  - {row.get('run_id', '')}: {row.get('status', '')} {row.get('skip_reason', '')}".rstrip()
        )

    report_lines = [
        "# Protocol V14 几何感知 Trace Bridge 跨构造验证报告",
        "",
        "## 1. 研究问题",
        "本轮不比较更大的模型，而是检查普通 RGB 式 bridge 是否因为忽略地震观测几何、邻道关系和整炮上下文，导致预训练 backbone 的优势不能稳定转化为跨构造表现。",
        "",
        "## 2. Geometry Provenance",
        f"- geometry provenance: {geometry.get('geometry_provenance', 'UNAVAILABLE')}",
        f"- 编码口径: {geometry.get('encoding_mode', '未说明')}",
        "",
        "## 3. Bridge 逐层设计",
        "- B0: 原始多通道波形信息",
        "- B1: 加入几何编码",
        "- B2: 加入邻道与整炮上下文",
        "- B3: 加入多尺度与频带条件",
        "",
        "## 4. 当前运行完成度",
        f"- 总 run 数: {summary.get('run_count', len(per_run))}",
        f"- 成功 run 数: {summary.get('success', 0)}",
        f"- 未成功 run 数: {summary.get('unsuccessful_count', len(unsuccessful))}",
        f"- 输出不完整 run 数: {summary.get('incomplete_output_count', len(incomplete))}",
        f"- 已汇总记录数: {len(per_run)}",
        f"- bootstrap 记录数: {len(bootstrap_rows)}",
        "",
        "## 5. 跨构造绝对指标",
        f"- 聚合条目数: {len(aggregate)}",
        "",
        "## 6. DINOv2 的 incremental bridge gain",
        f"- gain 行数: {_count_prefix(gain, 'M3')}",
        "",
        "## 7. NCS2D 的 incremental bridge gain",
        f"- gain 行数: {_count_prefix(gain, 'M6')}",
        "",
        "## 8. 未成功结果",
    ]
    if unsuccessful_examples:
        report_lines.extend(unsuccessful_examples)
    else:
        report_lines.append("- 当前未发现状态级未成功 run。")
    report_lines.extend(
        [
            "",
            "## 9. 输出完整性提醒",
        ]
    )
    if incomplete_examples:
        report_lines.extend(incomplete_examples)
    else:
        report_lines.append("- 当前未发现输出契约缺口。")
    report_lines.extend(
        [
            "",
            "## 10. 地震域预训练是否更依赖物理组织",
            "当前报告仅做方向性审计，不得声称 NCS2D 已证明优于 DINOv2。",
            "",
            "## 11. 少炮、缺道、噪声鲁棒性",
            "robustness 为 evaluation-only；若相关结果文件未完成或当前目录没有可复评 checkpoint，则不能扩展解释。",
            "",
            "## 12. 结果解释",
            "geometry encoder 的作用是告诉模型这条 trace 来自哪个炮、哪个接收器、对应多大偏移距以及所处时间位置；trace neighborhood 让模型看到邻近接收器的相关波形；shot global context 保留整炮记录的总体照明与能量分布；multiscale/frequency 条件帮助区分低频背景趋势和高频局部变化。这些都不是波动方程物理约束，只是更合理地组织观测信息。",
            "",
            "## 13. 结论等级",
            "当前仅允许使用“一致的方向性证据”“部分或混合证据”“未形成一致证据”三类表述。",
            "",
            "## 14. 局限性",
            "- CPU-only",
            "- 200/50/50 小样本",
            "- OpenFWI subset",
            "- 仅三种 cross-family transfer",
            "- mean aggregation",
            "- 无 PDE consistency loss",
            "- canonical reconstructed geometry 不等于真实采集几何",
            "- 不构成标准基准级结论",
            "- 不构成工程应用级性能",
            "",
            "## 15. Claims",
            "- 不能声称 geometry-aware trace bridge 已证明提升 FWI 泛化能力",
            "- 不能声称 canonical reconstructed geometry 等同真实采集几何",
            "- 不能声称地震域预训练已优于自然图像预训练",
            "- 当前 figures 目录仅对已有结果作可视化整理，不改变结论等级",
            "",
            "CPU 小样本统一协议；结果用于检验方向性证据，不构成标准基准级结论。",
        ]
    )

    report_path = protocol_root / "protocol_v14_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    claims_path = protocol_root / "protocol_v14_claims_and_limitations.md"
    claims_path.write_text(
        "\n".join(
            [
                "# Protocol V14 Claims and Limitations",
                "",
                "## Can Claim",
                "- 已在固定 backbone、decoder、训练预算和目标测试隔离条件下进行 geometry-aware trace bridge 消融。",
                "- 已区分真实采集几何、规范化索引几何和不可用状态。",
                "- 可基于当前 run 目录统计成功、未成功和输出不完整结果。",
                "",
                "## Cannot Claim",
                "- 不能声称 geometry-aware trace bridge 已证明提升 FWI 泛化能力。",
                "- 不能声称 canonical reconstructed geometry 等同真实采集几何。",
                "- 不能声称地震域预训练已优于自然图像预训练。",
                "- 当前结果不属于标准基准级结论。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"report_path": str(report_path), "claims_path": str(claims_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    print(json.dumps(report_protocol_v14_geometry_aware_trace_bridge(root=args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
