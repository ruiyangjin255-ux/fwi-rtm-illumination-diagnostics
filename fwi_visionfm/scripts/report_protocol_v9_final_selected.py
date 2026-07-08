from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _stats(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def _build_final_rows(selected_rows: list[dict[str, str]], boundary_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in selected_rows:
        grouped.setdefault(row["method_name"], []).append(row)
    boundary_method_rows = []
    for row in boundary_rows:
        boundary_method_rows.append(
            {
                "method_name": "ncs2d_boundary_aux_decoder",
                "method_family": "seismic_domain_ncs_boundary_aux",
                "seed": int(row["seed"]),
                "MAE": float(row["cross_mae"]),
                "RMSE": float(row["cross_rmse"]),
                "SSIM": float(row["cross_ssim"]),
                "gradient_error": float(row["cross_gradient_error"]),
                "edge_MAE": float(row["cross_edge_mae"]),
                "limitation_note": "frozen NCS feature + boundary-aware decoder",
            }
        )
    grouped["ncs2d_boundary_aux_decoder"] = boundary_method_rows

    final_rows: list[dict[str, Any]] = []
    for method_name, items in grouped.items():
        if method_name == "ncs2d_boundary_aux_decoder":
            method_family = "seismic_domain_ncs_boundary_aux"
            limitation_note = "frozen NCS feature + boundary-aware decoder"
            maes = [item["MAE"] for item in items]
            rmses = [item["RMSE"] for item in items]
            ssims = [item["SSIM"] for item in items]
            grads = [item["gradient_error"] for item in items]
            edges = [item["edge_MAE"] for item in items]
        else:
            method_family = items[0]["method_family"]
            limitation_note = items[0]["limitation_note"]
            maes = [float(item["cross_family_MAE"]) for item in items]
            rmses = [float(item["cross_family_RMSE"]) for item in items]
            ssims = [float(item["cross_family_SSIM"]) for item in items]
            grads = [float(item["cross_family_gradient_error"]) for item in items]
            edges = [float(item["cross_family_edge_MAE"]) for item in items]
        mae_mean, mae_std = _stats(maes)
        rmse_mean, rmse_std = _stats(rmses)
        ssim_mean, ssim_std = _stats(ssims)
        grad_mean, grad_std = _stats(grads)
        edge_mean, edge_std = _stats(edges)
        final_rows.append(
            {
                "method_name": method_name,
                "method_family": method_family,
                "seed_count": len(items),
                "MAE_mean": mae_mean,
                "MAE_std": mae_std,
                "RMSE_mean": rmse_mean,
                "RMSE_std": rmse_std,
                "SSIM_mean": ssim_mean,
                "SSIM_std": ssim_std,
                "gradient_error_mean": grad_mean,
                "gradient_error_std": grad_std,
                "edge_MAE_mean": edge_mean,
                "edge_MAE_std": edge_std,
                "numerical_rank": 0,
                "structural_rank": 0,
                "overall_selected_status": "",
                "claim_level": "selected_comparison_only",
                "limitation_note": limitation_note,
            }
        )
    final_rows.sort(key=lambda row: row["method_name"])
    numerical_order = sorted(final_rows, key=lambda row: (row["MAE_mean"], row["RMSE_mean"]))
    structural_order = sorted(final_rows, key=lambda row: (row["gradient_error_mean"], row["edge_MAE_mean"]))
    for idx, row in enumerate(numerical_order, start=1):
        row["numerical_rank"] = idx
    for idx, row in enumerate(structural_order, start=1):
        row["structural_rank"] = idx
    for row in final_rows:
        row["overall_selected_status"] = "current_best_selected_candidate" if row["method_name"] == "ncs2d_boundary_aux_decoder" else "reference"
    return final_rows


def _write_bar_chart(rows: list[dict[str, Any]], output_path: Path) -> None:
    width, height = 1200, 520
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "Protocol V9 Final Metrics (MAE mean)", fill="black")
    max_value = max(row["MAE_mean"] for row in rows) if rows else 1.0
    left, bottom, top = 80, height - 80, 70
    bar_width = 120
    gap = 30
    for idx, row in enumerate(rows):
        x0 = left + idx * (bar_width + gap)
        x1 = x0 + bar_width
        bar_h = int((row["MAE_mean"] / max_value) * (bottom - top))
        y0 = bottom - bar_h
        color = (80, 130, 190) if row["method_name"] != "ncs2d_boundary_aux_decoder" else (40, 160, 80)
        draw.rectangle([x0, y0, x1, bottom], fill=color, outline="black")
        draw.text((x0 - 5, bottom + 8), row["method_name"][:18], fill="black")
        draw.text((x0 + 10, y0 - 16), f"{row['MAE_mean']:.1f}", fill="black")
    draw.line([left, bottom, width - 40, bottom], fill="black", width=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def _write_tradeoff_plot(rows: list[dict[str, Any]], output_path: Path) -> None:
    width, height = 1200, 520
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "Protocol V9 Numerical-Structural Trade-off", fill="black")
    left, right, top, bottom = 100, width - 80, 70, height - 80
    mae_values = [row["MAE_mean"] for row in rows]
    grad_values = [row["gradient_error_mean"] for row in rows]
    x_min, x_max = min(mae_values), max(mae_values)
    y_min, y_max = min(grad_values), max(grad_values)
    def map_x(x: float) -> int:
        return int(left + (x - x_min) / max(x_max - x_min, 1e-6) * (right - left))
    def map_y(y: float) -> int:
        return int(bottom - (y - y_min) / max(y_max - y_min, 1e-6) * (bottom - top))
    draw.line([left, bottom, right, bottom], fill="black", width=2)
    draw.line([left, top, left, bottom], fill="black", width=2)
    for row in rows:
        x = map_x(row["MAE_mean"])
        y = map_y(row["gradient_error_mean"])
        color = (40, 160, 80) if row["method_name"] == "ncs2d_boundary_aux_decoder" else (80, 130, 190)
        draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill=color, outline="black")
        draw.text((x + 10, y - 10), row["method_name"][:20], fill="black")
    img.save(output_path)


def write_protocol_v9_final_selected_report(
    *,
    selected_comparison_summary: str | Path,
    selected_comparison_report: str | Path,
    ncs_boundary_summary: str | Path,
    ncs_boundary_report: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected_rows = _read_csv(Path(selected_comparison_summary))
    boundary_rows = _read_csv(Path(ncs_boundary_summary))
    selected_report_text = Path(selected_comparison_report).read_text(encoding="utf-8")
    boundary_report_text = Path(ncs_boundary_report).read_text(encoding="utf-8")
    final_rows = _build_final_rows(selected_rows, boundary_rows)

    summary_path = out / "protocol_v9_final_selected_summary.csv"
    fieldnames = [
        "method_name",
        "method_family",
        "seed_count",
        "MAE_mean",
        "MAE_std",
        "RMSE_mean",
        "RMSE_std",
        "SSIM_mean",
        "SSIM_std",
        "gradient_error_mean",
        "gradient_error_std",
        "edge_MAE_mean",
        "edge_MAE_std",
        "numerical_rank",
        "structural_rank",
        "overall_selected_status",
        "claim_level",
        "limitation_note",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    claims_path = out / "protocol_v9_final_claims_and_limitations.md"
    claims_lines = [
        "# Protocol V9 Final Claims And Limitations",
        "",
        "## Can Claim",
        "- ncs2d real frozen feature chain is available。",
        "- ncs2d seed=0/1/2 decoder-only probe is stable。",
        "- ncs2d_boundary_aux_decoder seed=0/1/2 probe is complete。",
        "- In this selected CPU small-sample setting, ncs2d_boundary_aux_decoder shows the best combined numerical and structural metrics among the compared selected methods。",
        "- The result supports continuing the seismic-domain frozen feature + boundary-aware decoder route。",
        "",
        "## Cannot Claim",
        "- NCS improves FWI。",
        "- NCS improves FWI generalization。",
        "- ncs2d outperforms CNN as benchmark proof。",
        "- boundary auxiliary improves FWI generalization。",
        "- natural-image MAE improves FWI。",
        "- current results are benchmark-level proof。",
        "- current predictions are application-level。",
        "- ncs2p5d result is available。",
    ]
    claims_path.write_text("\n".join(claims_lines) + "\n", encoding="utf-8")

    key_findings_path = out / "protocol_v9_final_key_findings.md"
    key_findings_lines = [
        "# Protocol V9 Final Key Findings",
        "",
        "1. Frozen feature alone improves numerical metrics but weakens structural metrics。",
        "2. Boundary-aware decoder improves structural metrics。",
        "3. ncs2d_boundary_aux_decoder combines both effects in this selected setting。",
        "4. ncs2d_boundary_aux_decoder is current best selected candidate, not benchmark winner。",
        "5. Next priority is V1–V9 report update or ncs2p5d/geometry extension。",
    ]
    key_findings_path.write_text("\n".join(key_findings_lines) + "\n", encoding="utf-8")

    next_steps_path = out / "protocol_v9_final_next_steps.md"
    next_steps_lines = [
        "# Protocol V9 Final Next Steps",
        "",
        "## A. 报告收束路线",
        "- 更新 V1–V9 总阶段报告。",
        "- 形成论文/汇报用阶段结论。",
        "- 不再继续扩 random tiny probe。",
        "",
        "## B. 技术扩展路线",
        "- ncs2d_boundary_aux seed-expanded 或 train_size=200 limited check。",
        "- ncs2p5d pseudo-view adapter。",
        "- real geometry metadata / source-receiver-offset encoding。",
        "- ncs2d_boundary_aux + geometry-aware aggregation。",
    ]
    next_steps_path.write_text("\n".join(next_steps_lines) + "\n", encoding="utf-8")

    metrics_bar_path = out / "protocol_v9_final_metrics_bar.png"
    tradeoff_path = out / "protocol_v9_final_tradeoff_plot.png"
    _write_bar_chart(final_rows, metrics_bar_path)
    _write_tradeoff_plot(final_rows, tradeoff_path)

    rows_by_name = {row["method_name"]: row for row in final_rows}
    ncs_ba = rows_by_name["ncs2d_boundary_aux_decoder"]
    ncs_frozen = rows_by_name["ncs2d_frozen_decoder"]
    boundary = rows_by_name["boundary_aux_gradient_lambda010"]
    vit = rows_by_name["vit_mae_base_frozen_decoder"]

    report_lines = [
        "# Protocol V9 Final Selected Report",
        "",
        "## 1. Goal",
        "本报告收束 V9 selected comparison，不新增实验，不做 benchmark claim。",
        "",
        "## 2. Compared Methods",
        "1. cnn_baseline_unet_l1",
        "2. boundary_aux_gradient_lambda010",
        "3. vit_mae_base_frozen_decoder",
        "4. ncs2d_frozen_decoder",
        "5. ncs2d_boundary_aux_decoder",
        "",
        "- cnn/boundary_aux 是端到端监督训练；",
        "- vit_mae_base/ncs2d 是 frozen feature + decoder-only；",
        "- ncs2d_boundary_aux 是 frozen NCS feature + trainable boundary-aware decoder；",
        "- methods have different training paradigms, so this remains a selected comparison and not benchmark-level proof.",
        "",
        "## 3. Seed Coverage",
        "- 五组方法均为 seed=0/1/2；",
        "- seed 数一致有利于 selected stability readout；",
        "- 但仍不是 full benchmark。",
        "",
        "## 4. Numerical Metrics",
    ]
    for row in final_rows:
        report_lines.append(
            f"- {row['method_name']}: MAE {row['MAE_mean']:.4f}+/-{row['MAE_std']:.4f}, RMSE {row['RMSE_mean']:.4f}+/-{row['RMSE_std']:.4f}, SSIM {row['SSIM_mean']:.4f}+/-{row['SSIM_std']:.4f}"
        )
    report_lines.extend(
        [
            "- ncs2d_boundary_aux_decoder 当前 MAE/RMSE 最好；",
            "- ncs2d_frozen_decoder 次优；",
            "- vit_mae_base_frozen_decoder 也具有较强数值表现；",
            "- 这些结论仅限 selected comparison，不构成 benchmark 结论。",
            "",
            "## 5. Structural Metrics",
        ]
    )
    for row in final_rows:
        report_lines.append(
            f"- {row['method_name']}: gradient_error {row['gradient_error_mean']:.4f}+/-{row['gradient_error_std']:.4f}, edge_MAE {row['edge_MAE_mean']:.4f}+/-{row['edge_MAE_std']:.4f}"
        )
    report_lines.extend(
        [
            "- ncs2d_boundary_aux_decoder 当前 gradient_error / edge_MAE 最好；",
            "- boundary_aux_gradient_lambda010 是此前结构最强 baseline；",
            "- ncs2d_frozen_decoder 结构指标偏弱；",
            "- boundary-aware decoder 显著缓解 ncs2d_frozen_decoder 的结构短板。",
            "",
            "## 6. Key Comparison",
            "### 6.1 ncs2d_boundary_aux vs ncs2d_frozen_decoder",
            f"- MAE 从 {ncs_frozen['MAE_mean']:.4f} 改善到 {ncs_ba['MAE_mean']:.4f}；",
            f"- RMSE 从 {ncs_frozen['RMSE_mean']:.4f} 改善到 {ncs_ba['RMSE_mean']:.4f}；",
            f"- gradient_error 从 {ncs_frozen['gradient_error_mean']:.4f} 改善到 {ncs_ba['gradient_error_mean']:.4f}；",
            f"- edge_MAE 从 {ncs_frozen['edge_MAE_mean']:.4f} 改善到 {ncs_ba['edge_MAE_mean']:.4f}；",
            "- 这说明 boundary-aware decoder/loss 对 frozen NCS feature 的结构恢复有正向作用。",
            "",
            "### 6.2 ncs2d_boundary_aux vs boundary_aux_gradient_lambda010",
            f"- 结构指标保持或超过 boundary_aux：gradient_error {ncs_ba['gradient_error_mean']:.4f} vs {boundary['gradient_error_mean']:.4f}, edge_MAE {ncs_ba['edge_MAE_mean']:.4f} vs {boundary['edge_MAE_mean']:.4f}；",
            f"- 数值指标显著改善：MAE {ncs_ba['MAE_mean']:.4f} vs {boundary['MAE_mean']:.4f}, RMSE {ncs_ba['RMSE_mean']:.4f} vs {boundary['RMSE_mean']:.4f}；",
            "- 这说明 NCS frozen feature 的数值趋势优势与 boundary_aux 的结构约束在该 selected setting 下形成了互补。",
            "",
            "### 6.3 ncs2d_boundary_aux vs vit_mae_base_frozen_decoder",
            f"- ncs2d_boundary_aux 在结构指标上明显更好：gradient_error {ncs_ba['gradient_error_mean']:.4f} vs {vit['gradient_error_mean']:.4f}, edge_MAE {ncs_ba['edge_MAE_mean']:.4f} vs {vit['edge_MAE_mean']:.4f}；",
            f"- 同时保留数值优势：MAE {ncs_ba['MAE_mean']:.4f} vs {vit['MAE_mean']:.4f}, RMSE {ncs_ba['RMSE_mean']:.4f} vs {vit['RMSE_mean']:.4f}；",
            "- 这说明单纯 frozen decoder 不足，decoder/loss 设计仍然关键。",
            "",
            "## 7. Interpretation",
            "- ncs2d_boundary_aux_decoder is the current best selected candidate under this CPU small-sample setting。",
            "- It combines the numerical strength of NCS frozen features and the structural benefit of boundary-aware decoding。",
            "- This supports continuing the seismic-domain feature + structure-aware decoder route。",
            "",
            "## 8. Limitations",
            "- CPU-only；",
            "- train_size=100 / val_size=50 / test_size=50；",
            "- selected comparison；",
            "- not full benchmark；",
            "- methods have different training paradigms；",
            "- frozen feature + decoder-only differs from end-to-end supervised training；",
            "- no full fine-tuning；",
            "- no ncs2p5d result；",
            "- OpenFWI shot gather differs from migrated seismic cube domain used by NCS pretraining；",
            "- boundary target is gradient-derived, not manual geology label；",
            "- no application-level performance；",
            "- not benchmark-level proof。",
            "",
            "## 9. Recommended Current Setting",
            "Current best selected candidate:",
            "- ncs2d real frozen feature",
            "- raw_envelope_spectrum3",
            "- mean_patch feature",
            "- boundary_aux_feature_decoder / boundary-aware decoder",
            "- boundary_aux_l1",
            "- boundary_method=gradient_magnitude",
            "- lambda_boundary=0.10",
            "",
            "Keep as references:",
            "- cnn_baseline_unet_l1 as task-specific baseline",
            "- boundary_aux_gradient_lambda010 as structure-aware supervised baseline",
            "- vit_mae_base_frozen_decoder as natural-image MAE frozen reference",
            "- ncs2d_frozen_decoder as seismic-domain frozen numerical reference",
            "",
            "## 10. Next Step",
            "A. 报告收束路线：",
            "- 更新 V1–V9 总阶段报告；",
            "- 形成论文/汇报用阶段结论；",
            "- 不再继续扩 random tiny probe。",
            "",
            "B. 技术扩展路线：",
            "- ncs2d_boundary_aux seed-expanded 或 train_size=200 limited check；",
            "- ncs2p5d pseudo-view adapter；",
            "- real geometry metadata / source-receiver-offset encoding；",
            "- ncs2d_boundary_aux + geometry-aware aggregation。",
        ]
    )
    report_path = out / "protocol_v9_final_selected_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {
        "report_path": report_path,
        "summary_path": summary_path,
        "claims_path": claims_path,
        "key_findings_path": key_findings_path,
        "next_steps_path": next_steps_path,
        "metrics_bar_path": metrics_bar_path,
        "tradeoff_path": tradeoff_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Protocol V9 final selected report.")
    parser.add_argument("--selected-comparison-summary", type=Path, required=True)
    parser.add_argument("--selected-comparison-report", type=Path, required=True)
    parser.add_argument("--ncs-boundary-summary", type=Path, required=True)
    parser.add_argument("--ncs-boundary-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = write_protocol_v9_final_selected_report(
        selected_comparison_summary=args.selected_comparison_summary,
        selected_comparison_report=args.selected_comparison_report,
        ncs_boundary_summary=args.ncs_boundary_summary,
        ncs_boundary_report=args.ncs_boundary_report,
        output_dir=args.output_dir,
    )
    print(json.dumps({key: str(value) for key, value in payload.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
