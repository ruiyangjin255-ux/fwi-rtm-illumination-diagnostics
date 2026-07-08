from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from fwi_visionfm.scripts.summarize_protocol_v2 import write_summary


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _table(rows: list[dict[str, Any]], limit: int = 20) -> str:
    cols = ["source_family", "target_family", "model_name", "bridge", "seed", "cross_family_MAE", "cross_family_RMSE", "cross_family_SSIM", "cross_family_gradient_error", "status"]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def _read_bridge_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _skipped_table(rows: list[dict[str, Any]]) -> str:
    skipped = [row for row in rows if str(row.get("status", "")).upper().startswith("SKIPPED")]
    if not skipped:
        return "No skipped runs were recorded."
    cols = ["source_family", "target_family", "model_name", "bridge", "seed", "status", "skip_reason"]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in skipped:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def _bridge_comparison_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "当前没有可用的 bridge 对比结果。"
    cols = [
        "source_family",
        "target_family",
        "model_name",
        "seed",
        "delta_MAE",
        "delta_RMSE",
        "delta_SSIM",
        "delta_gradient_error",
        "conclusion",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        conclusion = "numerical gain without established structural gain" if row.get("numerical_gain_without_structural_gain") == "True" else (
            "numerical and structural gain" if row.get("numerical_gain") == "True" and row.get("structural_gradient_gain") == "True" else "no clear bridge advantage"
        )
        lines.append(
            "| "
            + " | ".join(
                str(
                    {
                        "source_family": row.get("source_family", ""),
                        "target_family": row.get("target_family", ""),
                        "model_name": row.get("model_name", ""),
                        "seed": row.get("seed", ""),
                        "delta_MAE": row.get("delta_cross_MAE", ""),
                        "delta_RMSE": row.get("delta_cross_RMSE", ""),
                        "delta_SSIM": row.get("delta_cross_SSIM", ""),
                        "delta_gradient_error": row.get("delta_cross_gradient_error", ""),
                        "conclusion": conclusion,
                    }[col]
                )
                for col in cols
            )
            + " |"
        )
    return "\n".join(lines)


def _generalization_notes(rows: list[dict[str, Any]]) -> list[str]:
    flagged = [row for row in rows if row.get("numerical_gain_without_structural_gain") == "True"]
    if not flagged:
        return ["- 当前可用 bridge 对比中，未检测到“numerical gain without established structural gain”。"]
    notes = []
    for row in flagged:
        notes.append(
            f"- {row['source_family']} -> {row['target_family']} / {row['model_name']} / seed {row['seed']}: numerical gain without established structural gain."
        )
    return notes


def _bridge_conclusion(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "当前输出不包含成对 bridge 对比，不能据此判断 numerical gain vs structural gain。"
    flagged = [row for row in rows if row.get("numerical_gain_without_structural_gain") == "True"]
    if flagged:
        return "Scratch ViT with spectrogram bridge shows preliminary numerical gains under CPU small-sample protocol, but structural recovery improvement is not established."
    return "当前可用 bridge 对比未显示稳定的“numerical gain without established structural gain”模式。"


def _metric_space_summary(rows: list[dict[str, Any]]) -> str:
    spaces = sorted({row.get("metric_space", "") for row in rows if row.get("metric_space")})
    if not spaces:
        return "unknown"
    if len(spaces) == 1:
        return spaces[0]
    return ", ".join(spaces)


def _probe_note(rows: list[dict[str, Any]]) -> list[str]:
    if len(rows) != 1:
        return []
    row = rows[0]
    if not str(row.get("model_name", "")).startswith("dinov2"):
        return []
    return [
        "",
        "本报告对应 single-pair single-seed DINOv2-LoRA probe，不是正式 benchmark。",
    ]


def build_report(root: str | Path) -> Path:
    output_root = Path(root)
    paths = write_summary(output_root)
    rows = _read_rows(paths["summary"])
    bridge_rows = _read_bridge_rows(paths["bridge_comparison"])
    metric_space = _metric_space_summary(rows)
    report_path = output_root / "protocol_v2_report.md"
    lines = [
        "# Protocol V2 小规模基准报告",
        "",
        "## 1. 目标",
        "本协议是 CPU 小样本基准，不是大规模训练，也不是最终 DINOv2 benchmark。它用于检验 VisionFM 跨模态迁移是否改善端到端 FWI 的跨 family 泛化。",
        "",
        "Protocol V2 real-training smoke has been completed under CPU small-sample settings.",
        "",
        "## 2. 数据协议",
        "split 使用 source-family train/val/in-family test 与 target-family cross-family test。归一化统计量仅来自 source-family train split。默认协议是 train=500, val=100, in-family test=100, cross-family test=100, seeds 0/1/2。",
        "",
        f"当前结果指标空间：{metric_space}。",
        "若 prediction npz 与 metrics json 显式写出 `physical_velocity`，则表示评估已落在物理速度空间；否则保持 `normalized_tensor` 口径。",
        *_probe_note(rows),
        "",
        "## 3. 模型矩阵",
        "- CNN baseline",
        "- random ViT / ViT tiny scratch",
        "- DINOv2 frozen",
        "- DINOv2-LoRA",
        "- spectrogram-DINOv2 / spectrogram-DINOv2-LoRA",
        "",
        "## 4. 指标",
        "MAE 与 RMSE 反映数值误差。SSIM 与 gradient error 反映结构恢复。PSNR 作为图像式重建质量指标保留。若可用，edge MAE 仍保存在每个 run 的 JSON 中。",
        f"本报告第 5-7 节的汇总解释基于 `{metric_space}` 指标空间。",
        "",
        "## 5. 主要结果",
        _table(rows),
        "",
        "跳过的 runs：",
        "",
        _skipped_table(rows),
        "",
        "## 6. 泛化差距",
        "`protocol_v2_summary.csv` 中的 generalization gap 使用 cross-family 减去 in-family 的 MAE、RMSE 与 gradient error。",
        "",
        "## 7. Numerical Gain vs Structural Gain",
        "这里按同一个 source_family、target_family、model_name、seed 比较 `raw_repeat3` 与 `raw_spectrogram`。",
        "",
        *(_generalization_notes(bridge_rows)),
        "",
        _bridge_comparison_table(bridge_rows),
        "",
        _bridge_conclusion(bridge_rows),
        "",
        "## 8. 局限性",
        "- CPU 小样本协议",
        "- 默认 3 epoch，本轮 smoke 常为 1 epoch",
        "- real DINOv2 仍可能因默认跳过策略或环境限制而未评估",
        "- 不能作为最终大规模 DINOv2 benchmark",
        "",
        "## 9. 下一步",
        "- train=2000",
        "- GPU runs",
        "- more families",
        "- attention and UMAP interpretability analysis",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write protocol v2 Markdown report.")
    parser.add_argument("--root", type=Path, default=Path("outputs/protocol_v2_small"))
    return parser.parse_args()


def main() -> None:
    path = build_report(parse_args().root)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
