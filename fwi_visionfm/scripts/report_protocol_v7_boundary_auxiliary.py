from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fwi_visionfm.scripts.run_protocol_v7_boundary_auxiliary_smoke import summarize_protocol_v7_runs


def write_protocol_v7_boundary_auxiliary_report(root: str | Path) -> Path:
    output_root = Path(root)
    summary_path = output_root / "protocol_v7_boundary_auxiliary_summary.csv"
    if not summary_path.exists():
        summary_path = summarize_protocol_v7_runs(output_root)
    rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8")))
    lines = [
        "# Protocol V7 Boundary Auxiliary Smoke Report",
        "",
        "## 1. Goal",
        "本轮只验证 boundary auxiliary head 是否可进入真实训练链路，并初步观察结构指标方向性，不做 benchmark claim。",
        "",
        "## 2. Runs",
    ]
    for row in rows:
        lines.append(f"- {row['run_id']}: decoder={row['decoder']}, loss={row['loss']}, status={row['status']}")
    lines.extend(
        [
            "",
            "## 3. Boundary Target Check",
            "- boundary target 由 velocity_true 自动生成。",
            "- gradient_magnitude 与 sobel 两种方法都进入了 smoke 配置。",
            "- 如果 boundary 分支成功，predictions_*.npz 中会保存 boundary_pred / boundary_target / boundary_error_map。",
            "- boundary target 通过 boundary_targets.py 归一化到 [0, 1]。",
            "",
            "## 4. Boundary Decoder Check",
            "- boundary_aux_unet 若成功，表示真实训练链路中同时输出 velocity 与 boundary。",
            "- boundary_prediction_grid.png 的存在用于确认 boundary 预测可视化链路可用。",
            "",
            "## 5. Metrics",
        ]
    )
    for row in rows:
        lines.append(
            f"- {row['run_id']}: val(MAE={row['val_MAE']}, RMSE={row['val_RMSE']}, SSIM={row['val_SSIM']}), "
            f"cross-family(MAE={row['cross_family_MAE']}, RMSE={row['cross_family_RMSE']}, SSIM={row['cross_family_SSIM']}, "
            f"gradient_error={row['cross_family_gradient_error']}, edge_MAE={row['cross_family_edge_MAE']}), boundary_val_L1={row['boundary_val_L1']}"
        )
    lines.extend(
        [
            "- 本节只允许做 smoke-scale directional observation，不作为“boundary auxiliary 已经提升 FWI 泛化能力”的证据。",
            "",
            "## 6. Limitations",
            "- CPU small-sample",
            "- smoke-scale only",
            "- not benchmark-level proof",
            "- no DINOv2/SAM/NCS in this run",
            "- boundary target is derived from velocity gradients, not manually labeled geology",
            "- only seed=0",
            "",
            "## 7. Next Step",
            "- 如果 boundary_aux_unet 成功并且 gradient_error / edge_MAE 有方向性改善，后续可扩大到 seed=0/1/2。",
            "- 如果结构指标无改善，优先检查 lambda_boundary、boundary_method 和 decoder capacity。",
            "- 如果 MAE/RMSE 明显变差，优先降低 lambda_boundary 到 0.01 或改为 weak boundary loss。",
        ]
    )
    report = output_root / "protocol_v7_boundary_auxiliary_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Protocol V7 boundary auxiliary report.")
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(write_protocol_v7_boundary_auxiliary_report(parse_args().root))


if __name__ == "__main__":
    main()
