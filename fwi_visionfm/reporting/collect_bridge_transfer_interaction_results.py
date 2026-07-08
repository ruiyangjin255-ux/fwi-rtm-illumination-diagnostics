from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_parameter_report(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    trainable_match = re.search(r"trainable parameters:\s*(\d+)", text)
    ratio_match = re.search(r"trainable ratio:\s*([0-9.]+)", text)
    return {
        "trainable_params": int(trainable_match.group(1)) if trainable_match else "NA",
        "trainable_ratio": float(ratio_match.group(1)) if ratio_match else "NA",
    }


def _collect_one(experiment_dir: Path, *, bridge_override: str | None = None) -> dict[str, Any]:
    config_path = experiment_dir / "config_resolved.json"
    if not config_path.exists():
        config_path = experiment_dir / "resolved_foundation_config.json"
    config = _read_json(config_path)
    metrics_in = _read_json(experiment_dir / "metrics_in_family_extended.json")
    metrics_cross = _read_json(experiment_dir / "metrics_cross_family_extended.json")
    params = (
        _parse_parameter_report(experiment_dir / "parameter_report.txt")
        if (experiment_dir / "parameter_report.txt").exists()
        else {"trainable_params": "NA", "trainable_ratio": "NA"}
    )
    bridge_feature_mode = bridge_override or config.get("bridge_feature_mode", "raw_repeat3")
    row = {
        "experiment": experiment_dir.name,
        "transfer_mode": config.get("transfer_mode", "NA"),
        "peft": config.get("peft_type", "NA"),
        "bridge_feature_mode": bridge_feature_mode,
        "pretrained": config.get("pretrained", "NA"),
        "in_family_mae": metrics_in.get("mae", "NA"),
        "in_family_rmse": metrics_in.get("rmse", "NA"),
        "in_family_psnr": metrics_in.get("psnr", "NA"),
        "in_family_ssim": metrics_in.get("ssim", "NA"),
        "in_family_edge_mae": metrics_in.get("edge_mae", "NA"),
        "in_family_laplacian_mae": metrics_in.get("laplacian_mae", "NA"),
        "in_family_horizon_gradient_mae": metrics_in.get("horizon_gradient_mae", "NA"),
        "in_family_vertical_gradient_mae": metrics_in.get("vertical_gradient_mae", "NA"),
        "cross_family_mae": metrics_cross.get("mae", "NA"),
        "cross_family_rmse": metrics_cross.get("rmse", "NA"),
        "cross_family_psnr": metrics_cross.get("psnr", "NA"),
        "cross_family_ssim": metrics_cross.get("ssim", "NA"),
        "cross_family_edge_mae": metrics_cross.get("edge_mae", "NA"),
        "cross_family_laplacian_mae": metrics_cross.get("laplacian_mae", "NA"),
        "cross_family_laplian_mae": metrics_cross.get("laplacian_mae", "NA"),
        "cross_family_horizon_gradient_mae": metrics_cross.get("horizon_gradient_mae", "NA"),
        "cross_family_vertical_gradient_mae": metrics_cross.get("vertical_gradient_mae", "NA"),
        "generalization_gap_mae": float(metrics_cross["mae"]) - float(metrics_in["mae"]),
        "generalization_gap_rmse": float(metrics_cross["rmse"]) - float(metrics_in["rmse"]),
        "trainable_params": params.get("trainable_params", "NA"),
        "trainable_ratio": params.get("trainable_ratio", "NA"),
        "output_dir": str(experiment_dir),
    }
    return row


def _format_delta(delta: float) -> str:
    if delta < 0:
        return f"improved by {abs(delta):.6f}"
    if delta > 0:
        return f"worse by {delta:.6f}"
    return "matched"


def _compare(row_a: dict[str, Any], row_b: dict[str, Any], key: str) -> str:
    return _format_delta(float(row_a[key]) - float(row_b[key]))


def _fieldnames() -> list[str]:
    return [
        "experiment",
        "transfer_mode",
        "peft",
        "bridge_feature_mode",
        "pretrained",
        "in_family_mae",
        "in_family_rmse",
        "in_family_psnr",
        "in_family_ssim",
        "in_family_edge_mae",
        "in_family_laplacian_mae",
        "in_family_horizon_gradient_mae",
        "in_family_vertical_gradient_mae",
        "cross_family_mae",
        "cross_family_rmse",
        "cross_family_psnr",
        "cross_family_ssim",
        "cross_family_edge_mae",
        "cross_family_laplacian_mae",
        "cross_family_laplian_mae",
        "cross_family_horizon_gradient_mae",
        "cross_family_vertical_gradient_mae",
        "generalization_gap_mae",
        "generalization_gap_rmse",
        "delta_cross_mae_vs_same_transfer_raw_repeat3",
        "delta_cross_rmse_vs_same_transfer_raw_repeat3",
        "delta_cross_edge_mae_vs_same_transfer_raw_repeat3",
        "delta_cross_laplacian_mae_vs_same_transfer_raw_repeat3",
        "delta_gap_mae_vs_same_transfer_raw_repeat3",
        "trainable_params",
        "trainable_ratio",
        "output_dir",
    ]


def _compute_transfer_deltas(rows: list[dict[str, Any]]) -> None:
    baselines = {
        row["transfer_mode"]: row
        for row in rows
        if row["bridge_feature_mode"] == "raw_repeat3"
    }
    for row in rows:
        baseline = baselines.get(row["transfer_mode"])
        if baseline is None:
            continue
        row["delta_cross_mae_vs_same_transfer_raw_repeat3"] = float(row["cross_family_mae"]) - float(baseline["cross_family_mae"])
        row["delta_cross_rmse_vs_same_transfer_raw_repeat3"] = float(row["cross_family_rmse"]) - float(baseline["cross_family_rmse"])
        row["delta_cross_edge_mae_vs_same_transfer_raw_repeat3"] = float(row["cross_family_edge_mae"]) - float(baseline["cross_family_edge_mae"])
        row["delta_cross_laplacian_mae_vs_same_transfer_raw_repeat3"] = float(row["cross_family_laplacian_mae"]) - float(baseline["cross_family_laplacian_mae"])
        row["delta_gap_mae_vs_same_transfer_raw_repeat3"] = float(row["generalization_gap_mae"]) - float(baseline["generalization_gap_mae"])


def _read_rows(experiment_specs: list[tuple[Path, str, str | None]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root, name, bridge_override in experiment_specs:
        experiment_dir = root / name
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Missing experiment directory: {experiment_dir}")
        rows.append(_collect_one(experiment_dir, bridge_override=bridge_override))
    rows.sort(key=lambda item: (str(item["transfer_mode"]), str(item["bridge_feature_mode"]), str(item["experiment"])))
    _compute_transfer_deltas(rows)
    return rows


def _summary_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| experiment | transfer_mode | bridge_feature_mode | in_family_mae | cross_family_mae | generalization_gap_mae | delta_cross_mae_vs_same_transfer_raw_repeat3 | trainable_ratio |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['transfer_mode']} | {row['bridge_feature_mode']} | "
            f"{row['in_family_mae']} | {row['cross_family_mae']} | {row['generalization_gap_mae']} | "
            f"{row['delta_cross_mae_vs_same_transfer_raw_repeat3']} | {row['trainable_ratio']} |"
        )
    return lines


def collect_bridge_transfer_interaction_results(
    *,
    scratch_root: str | Path,
    interaction_root: str | Path,
    output_root: str | Path,
    adapter_root: str | Path | None = None,
) -> dict[str, Any]:
    scratch_root = Path(scratch_root)
    interaction_root = Path(interaction_root)
    output_root = Path(output_root)
    adapter_root = Path(adapter_root) if adapter_root is not None else interaction_root
    output_root.mkdir(parents=True, exist_ok=True)

    experiment_specs = [
        (scratch_root, "scratch_bridge_raw_repeat3_3ep", None),
        (scratch_root, "scratch_bridge_raw_spectrogram_3ep", None),
        (interaction_root, "lora_bridge_raw_repeat3_3ep", None),
        (interaction_root, "lora_bridge_raw_spectrogram_3ep", None),
        (adapter_root, "timm_vit_tiny_adapter_openfwi_3ep", "raw_repeat3"),
        (interaction_root, "adapter_bridge_raw_spectrogram_3ep", None),
    ]
    rows = _read_rows(experiment_specs)
    if len(rows) != 6:
        raise ValueError(f"Expected 6 experiments, got {len(rows)}")

    csv_path = output_root / "summary_bridge_transfer_interaction_v2.csv"
    md_path = output_root / "summary_bridge_transfer_interaction_v2.md"
    report_path = output_root / "bridge_transfer_interaction_report_v2.md"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames())
        writer.writeheader()
        writer.writerows(rows)

    md_lines = ["# Bridge x Transfer Interaction Summary v2", "", *_summary_table(rows)]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    by_name = {row["experiment"]: row for row in rows}
    sr = by_name["scratch_bridge_raw_repeat3_3ep"]
    ss = by_name["scratch_bridge_raw_spectrogram_3ep"]
    lr = by_name["lora_bridge_raw_repeat3_3ep"]
    ls = by_name["lora_bridge_raw_spectrogram_3ep"]
    ar = by_name["timm_vit_tiny_adapter_openfwi_3ep"]
    a_s = by_name["adapter_bridge_raw_spectrogram_3ep"]

    best_cross = min(rows, key=lambda row: float(row["cross_family_mae"]))
    best_edge = min(rows, key=lambda row: float(row["cross_family_edge_mae"]))
    best_laplacian = min(rows, key=lambda row: float(row["cross_family_laplacian_mae"]))
    best_horizon = min(rows, key=lambda row: float(row["cross_family_horizon_gradient_mae"]))
    best_vertical = min(rows, key=lambda row: float(row["cross_family_vertical_gradient_mae"]))

    adapter_cross_improved = float(a_s["cross_family_mae"]) < float(ar["cross_family_mae"])

    report_lines = [
        "# Bridge x Transfer Interaction Report v2",
        "",
        "## 指标表",
        "",
        *_summary_table(rows),
        "",
        "## 问题回答",
        "",
        f"1. 在 scratch 下，raw_spectrogram 相比 raw_repeat3 的 in-family：MAE {_compare(ss, sr, 'in_family_mae')}；cross-family：MAE {_compare(ss, sr, 'cross_family_mae')}。",
        f"2. 在 LoRA 下，raw_spectrogram 相比 raw_repeat3 的 in-family：MAE {_compare(ls, lr, 'in_family_mae')}；cross-family：MAE {_compare(ls, lr, 'cross_family_mae')}。",
        f"3. 在 Adapter 下，raw_spectrogram 相比 raw_repeat3 的 in-family：MAE {_compare(a_s, ar, 'in_family_mae')}；cross-family：MAE {_compare(a_s, ar, 'cross_family_mae')}。",
        f"4. LoRA + raw_spectrogram 相比 scratch + raw_repeat3：cross-family MAE {_compare(ls, sr, 'cross_family_mae')}；相比 scratch + raw_spectrogram：{_compare(ls, ss, 'cross_family_mae')}；相比 LoRA + raw_repeat3：{_compare(ls, lr, 'cross_family_mae')}。",
        f"5. Adapter + raw_spectrogram 相比 scratch + raw_repeat3：cross-family MAE {_compare(a_s, sr, 'cross_family_mae')}；相比 scratch + raw_spectrogram：{_compare(a_s, ss, 'cross_family_mae')}；相比 Adapter + raw_repeat3：{_compare(a_s, ar, 'cross_family_mae')}。",
        "6. raw_spectrogram 的收益是否依赖 transfer mode：这轮必须分别看 scratch、LoRA、Adapter 三条线，而不能把 bridge 结论直接外推到所有 transfer mode。",
        "",
        "## 最优项",
        "",
        f"- best cross-family MAE: {best_cross['experiment']} ({best_cross['cross_family_mae']})",
        f"- best cross-family edge_mae: {best_edge['experiment']} ({best_edge['cross_family_edge_mae']})",
        f"- best cross-family laplacian_mae: {best_laplacian['experiment']} ({best_laplacian['cross_family_laplacian_mae']})",
        f"- best cross-family horizon_gradient_mae: {best_horizon['experiment']} ({best_horizon['cross_family_horizon_gradient_mae']})",
        f"- best cross-family vertical_gradient_mae: {best_vertical['experiment']} ({best_vertical['cross_family_vertical_gradient_mae']})",
        "",
        "## Guardrails",
        "",
        "- single seed",
        "- CPU",
        "- 3 epoch",
        "- 500/100/100/100",
        "- vit_tiny_patch16_224",
        "- not final DINOv2 conclusion",
    ]
    if adapter_cross_improved:
        report_lines.append("")
        report_lines.append("spectrogram bridge appears beneficial for PEFT target-family transfer under current setting.")
    else:
        report_lines.append("")
        report_lines.append("spectrogram bridge benefit is currently LoRA-specific and should not be generalized to all PEFT methods.")
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "row_count": len(rows),
        "csv_path": str(csv_path),
        "md_path": str(md_path),
        "report_path": str(report_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 bridge x transfer interaction 六组结果。")
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--interaction-root", required=True, type=Path)
    parser.add_argument("--adapter-root", type=Path, default=None)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = collect_bridge_transfer_interaction_results(
        scratch_root=args.scratch_root,
        interaction_root=args.interaction_root,
        adapter_root=args.adapter_root,
        output_root=args.output_root,
    )
    print(f"写出 interaction CSV: {result['csv_path']}")
    print(f"写出 interaction 报告: {result['report_path']}")


if __name__ == "__main__":
    main()
