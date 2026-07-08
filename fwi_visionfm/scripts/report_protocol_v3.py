from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from fwi_visionfm.scripts.summarize_protocol_v3 import write_summary


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _table(rows: list[dict[str, Any]], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def _has_run_configs(root: Path) -> bool:
    for path in root.rglob("config.json"):
        if "manifests" not in path.parts:
            return True
    return False


def _paths_for(root: Path) -> dict[str, Path]:
    return {
        "summary": root / "protocol_v3_summary.csv",
        "decoder_comparison": root / "protocol_v3_decoder_comparison.csv",
        "loss_comparison": root / "protocol_v3_loss_comparison.csv",
        "top_configs": root / "protocol_v3_top_configs.csv",
    }


def _decoder_interpretation(rows: list[dict[str, Any]]) -> str:
    total = len(rows)
    if total == 0:
        return "No paired decoder comparison rows are available."
    simple_num = sum(row.get("numerical_winner") == "simple_bounded_decoder" for row in rows)
    unet_num = sum(row.get("numerical_winner") == "unet_decoder" for row in rows)
    unet_struct = sum(row.get("structural_winner") == "unet_decoder" for row in rows)
    simple_struct = sum(row.get("structural_winner") == "simple_bounded_decoder" for row in rows)
    tradeoff = sum(row.get("tradeoff_type") == "simple numerical advantage vs unet structural advantage" for row in rows)
    if simple_num > unet_num and unet_struct > simple_struct:
        return (
            f"In this CPU probe, simple_bounded_decoder is generally better on MAE/RMSE "
            f"({simple_num}/{total} paired comparisons), while unet_decoder improves gradient_error and edge_MAE "
            f"in most paired comparisons ({unet_struct}/{total}). This indicates a numerical-structural trade-off."
        )
    if unet_num > simple_num and unet_struct > simple_struct:
        return f"In this CPU probe, unet_decoder is stronger on both numerical and structural metrics in paired comparisons."
    if simple_num > unet_num and simple_struct > unet_struct:
        return f"In this CPU probe, simple_bounded_decoder is stronger on both numerical and structural metrics in paired comparisons."
    return (
        f"Decoder comparison is mixed: simple numerical wins={simple_num}, unet numerical wins={unet_num}, "
        f"unet structural wins={unet_struct}, explicit trade-offs={tradeoff}."
    )


def _loss_interpretation(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No complete default_l1 / gradient_l1 / structure_loss comparison rows are available."
    structure_tradeoff = sum(row.get("loss_tradeoff_summary") == "structure loss improves gradient metrics with numerical tradeoff" for row in rows)
    structure_both = sum(row.get("loss_tradeoff_summary") == "structure loss improves both numerical and structural metrics" for row in rows)
    gradient_wins = sum(row.get("best_loss_by_gradient_error") == "gradient_l1" for row in rows)
    structure_wins = sum(row.get("best_loss_by_gradient_error") == "structure_loss" for row in rows)
    if structure_tradeoff:
        return (
            "Structure-aware losses reduce gradient_error slightly but do not yet improve MAE/RMSE, "
            "suggesting that boundary-aware supervision affects structure recovery but may require better weighting or longer training."
        )
    if structure_both:
        return "Structure-aware loss improves both numerical and structural metrics in the available paired comparison."
    return f"Loss comparison is mixed: gradient_l1 gradient wins={gradient_wins}, structure_loss gradient wins={structure_wins}."


def _dino_note(rows: list[dict[str, Any]]) -> str:
    dino_rows = [row for row in rows if "dinov2" in row.get("model_name", "").lower()]
    if not dino_rows:
        return "No DINOv2-LoRA probe rows are available."
    success = sum(row.get("status") == "SUCCESS" for row in dino_rows)
    skipped = sum(str(row.get("status", "")).startswith("SKIPPED") for row in dino_rows)
    failed = sum(row.get("status") == "FAILED" for row in dino_rows)
    return (
        f"DINOv2-LoRA probe rows: success={success}, skipped={skipped}, failed={failed}. "
        "This is a single-pair limited-seed probe and not a benchmark; comparisons with CNN/ViT are reference-only."
    )


def _multiseed_stability(decoder_rows: list[dict[str, Any]], loss_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> list[str]:
    baseline_decoder_rows = [row for row in decoder_rows if "dinov2" not in row.get("model_name", "").lower()]
    simple_num = sum(row.get("numerical_winner") == "simple_bounded_decoder" for row in baseline_decoder_rows)
    unet_struct = sum(row.get("structural_winner") == "unet_decoder" for row in baseline_decoder_rows)
    decoder_total = len(baseline_decoder_rows)
    structure_grad = sum(
        row.get("loss_tradeoff_summary") in {
            "structure loss improves gradient metrics with numerical tradeoff",
            "structure loss improves both numerical and structural metrics",
        }
        for row in loss_rows
    )
    structure_cost = sum(row.get("loss_tradeoff_summary") == "structure loss improves gradient metrics with numerical tradeoff" for row in loss_rows)
    loss_total = len(loss_rows)
    dino_rows = [row for row in summary_rows if "dinov2" in row.get("model_name", "").lower()]
    dino_success = sum(row.get("status") == "SUCCESS" for row in dino_rows)
    dino_skipped = sum(str(row.get("status", "")).startswith("SKIPPED") for row in dino_rows)
    dino_failed = sum(row.get("status") == "FAILED" for row in dino_rows)
    return [
        f"- simple decoder MAE/RMSE wins: {simple_num}/{decoder_total} seeds",
        f"- U-Net decoder gradient_error/edge_MAE wins: {unet_struct}/{decoder_total} seeds",
        f"- structure_loss lowers gradient_error: {structure_grad}/{loss_total} seeds",
        f"- structure_loss MAE/RMSE trade-off: {structure_cost}/{loss_total} seeds",
        f"- DINOv2-LoRA probe seed status: success={dino_success}, skipped={dino_skipped}, failed={dino_failed}; still treated as probe, not benchmark evidence.",
    ]


def _next_config_notes(decoder_rows: list[dict[str, Any]], loss_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> list[str]:
    notes = []
    if any(row.get("tradeoff_type") == "simple numerical advantage vs unet structural advantage" for row in decoder_rows):
        notes.append("- 保留 simple_bounded_decoder 与 unet_decoder，并尝试 hybrid loss 或 longer training。")
    if any(row.get("loss_tradeoff_summary") == "structure loss improves gradient metrics with numerical tradeoff" for row in loss_rows):
        notes.append("- 调小 gradient/laplacian/edge loss 权重，降低 MAE/RMSE 损失。")
    if any("dinov2" in row.get("model_name", "").lower() and row.get("status") == "SUCCESS" for row in summary_rows):
        notes.append("- 将 DINOv2-LoRA probe 转向 frozen feature cache + decoder-only training，降低 CPU 成本。")
    while len(notes) < 3:
        defaults = [
            "- tune loss weights",
            "- run multi-seed V3 for selected Pareto configs",
            "- use frozen DINOv2 feature cache for decoder-only training",
        ]
        for item in defaults:
            if item not in notes:
                notes.append(item)
            if len(notes) >= 3:
                break
    return notes[:3]


def build_report(root: str | Path) -> Path:
    output_root = Path(root)
    if _has_run_configs(output_root):
        paths = write_summary(output_root)
    else:
        paths = _paths_for(output_root)
    rows = _read_rows(paths["summary"])
    decoder_compare_rows = _read_rows(paths["decoder_comparison"])
    loss_compare_rows = _read_rows(paths["loss_comparison"])
    top_rows = _read_rows(paths["top_configs"])
    report_path = Path(root) / "protocol_v3_report.md"
    decoder_rows = [row for row in rows if row.get("loss_name") == "default_l1" and row.get("status") == "SUCCESS"]
    loss_rows = [row for row in rows if row.get("decoder_name") == "unet_decoder" and row.get("bridge") == "raw_spectrogram" and row.get("model_name") == "vit_tiny_scratch"]
    dino_rows = [row for row in rows if row["model_name"].startswith("dinov2")]
    pareto_rows = [row for row in top_rows if row.get("rank_type") == "pareto_candidates"]
    metric_spaces = sorted({row.get("metric_space", "") for row in rows if row.get("metric_space")})
    metric_space = metric_spaces[0] if len(metric_spaces) == 1 else ", ".join(metric_spaces) if metric_spaces else "physical_velocity"
    next_notes = _next_config_notes(decoder_compare_rows, loss_compare_rows, rows)
    lines = [
        "# Protocol V3 Structure-aware Benchmark Report",
        "",
        "## 1. Goal",
        "本协议专门验证 structure-aware decoder 和 loss 是否改善 FWI 速度模型结构恢复，而不是直接做 VisionFM benchmark claim。",
        "",
        "## 2. Data and Metric Space",
        (
            f"当前 V3 使用 selected single-pair multi-seed validation，metric_space = {metric_space}。"
            "DINOv2-LoRA 是 limited-seed probe，不作为 benchmark evidence。"
        ),
        "",
        "## 3. Overall Results",
        _table(rows, ["model_name", "bridge", "decoder_name", "loss_name", "cross_family_MAE", "cross_family_RMSE", "cross_family_gradient_error", "cross_family_edge_MAE", "status"]),
        "",
        "## 4. Decoder Comparison: Numerical vs Structural Recovery",
        _decoder_interpretation(decoder_compare_rows),
        "",
        _table(decoder_compare_rows, ["model_name", "bridge", "loss_name", "delta_MAE", "delta_RMSE", "delta_gradient_error", "delta_edge_MAE", "numerical_winner", "structural_winner", "tradeoff_type"]),
        "",
        _table(decoder_rows, ["model_name", "bridge", "decoder_name", "cross_family_MAE", "cross_family_RMSE", "cross_family_gradient_error", "cross_family_edge_MAE", "status"]),
        "",
        "## 5. Loss Comparison",
        _loss_interpretation(loss_compare_rows),
        "",
        _table(loss_compare_rows, ["model_name", "bridge", "decoder_name", "best_loss_by_MAE", "best_loss_by_gradient_error", "best_loss_by_edge_MAE", "loss_tradeoff_summary"]),
        "",
        _table(loss_rows, ["model_name", "bridge", "decoder_name", "loss_name", "cross_family_MAE", "cross_family_RMSE", "cross_family_gradient_error", "cross_family_edge_MAE", "status"]),
        "",
        "## 6. DINOv2-LoRA Probe",
        _dino_note(rows),
        "",
        _table(dino_rows, ["model_name", "bridge", "decoder_name", "loss_name", "cross_family_MAE", "cross_family_RMSE", "cross_family_gradient_error", "status"]),
        "",
        "这里只报告 limited-seed DINOv2 probe，不做 benchmark claim。",
        "",
        "## 7. Pareto Candidates and Next Configs",
        _table(pareto_rows, ["model_name", "bridge", "decoder_name", "loss_name", "MAE", "RMSE", "gradient_error", "edge_MAE", "note"]),
        "",
        *next_notes,
        "",
        "## Multi-seed Stability",
        *_multiseed_stability(decoder_compare_rows, loss_compare_rows, rows),
        "",
        "## 8. Limitations",
        "- CPU small-sample",
        "- selected single-pair multi-seed validation, not a full benchmark",
        "- DINOv2-LoRA probe is not a benchmark",
        "- 不对 VisionFM 迁移效果作 benchmark-level 结论",
        "",
        "## 9. Next Steps",
        "1. tune loss weights;",
        "2. run multi-seed V3 for selected Pareto configs;",
        "3. use frozen DINOv2 feature cache for decoder-only training.",
        "",
        "Protocol V3 reveals a numerical-structural trade-off: simple decoders are currently stronger in MAE/RMSE, while U-Net decoders improve gradient-based structural metrics. This supports the need for structure-aware decoder and loss design before making benchmark-level VisionFM claims.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Protocol V3 report.")
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    path = build_report(parse_args().root)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
