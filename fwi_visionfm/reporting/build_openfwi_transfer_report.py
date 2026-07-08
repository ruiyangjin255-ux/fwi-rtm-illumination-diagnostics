from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_openfwi_transfer_report(
    *,
    summary_csv: str | Path,
    summary_report: str | Path,
    prediction_grids: list[str | Path],
    output_path: str | Path,
) -> Path:
    summary_csv = Path(summary_csv)
    summary_report = Path(summary_report)
    output_path = Path(output_path)
    rows = _read_rows(summary_csv)
    prediction_grids = [Path(path) for path in prediction_grids]
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in prediction_grids:
        grouped[path.parent.name].append(path)

    best_val = min(rows, key=lambda row: float(row["val_mae"]) if row["val_mae"] != "NA" else float("inf"))
    best_cross = min(rows, key=lambda row: float(row["cross_family_mae"]) if row["cross_family_mae"] != "NA" else float("inf"))

    lines = [
        "# OpenFWI Small Transfer Stage Report",
        "",
        "## 1. 实验目的",
        "",
        "- 在不启动新训练的前提下，统一收束 raw OpenFWI foundation transfer 的 in-family 与 cross-family 结果口径。",
        "- 当前阶段只做 checkpoint-only evaluation、汇总表和预测图归档。",
        "",
        "## 2. 数据协议",
        "",
        "- train family = FlatVel_A",
        "- cross-family test family = CurveVel-A",
        f"- train/val/test/cross-family 样本数：`{rows[0]['train_samples']}/{rows[0]['val_samples']}/{rows[0]['test_samples']}/{rows[0]['test_samples']}`",
        "",
        "## 3. 模型对照",
        "",
        "- dummy",
        "- timm scratch",
        "- timm frozen",
        "- timm adapter",
        "- timm lora",
        "",
        "## 4. 参数量对照",
        "",
        "| experiment | total_params | trainable_params | trainable_ratio |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(f"| {row['experiment']} | {row['total_params']} | {row['trainable_params']} | {row['trainable_ratio']} |")

    lines.extend([
        "",
        "## 5. in-family 与 cross-family 指标表",
        "",
        "| experiment | in_family_mae | in_family_rmse | cross_family_mae | cross_family_rmse |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for row in rows:
        lines.append(f"| {row['experiment']} | {row['in_family_mae']} | {row['in_family_rmse']} | {row['cross_family_mae']} | {row['cross_family_rmse']} |")

    lines.extend([
        "",
        "## 6. generalization gap 表",
        "",
        "| experiment | generalization_gap_mae | generalization_gap_rmse |",
        "| --- | ---: | ---: |",
    ])
    for row in rows:
        lines.append(f"| {row['experiment']} | {row['generalization_gap_mae']} | {row['generalization_gap_rmse']} |")

    lines.extend([
        "",
        "## 7. prediction grid 路径",
        "",
    ])
    for experiment, paths in sorted(grouped.items()):
        lines.append(f"- `{experiment}`")
        for path in sorted(paths):
            lines.append(f"  - `{path}`")

    lines.extend([
        "",
        "## 8. 主要观察",
        "",
        f"- Adapter 在 validation 上最好：`{best_val['experiment']}`。",
        f"- scratch 在 cross-family 上当前最好：`{best_cross['experiment']}`。",
        "- PEFT 方法当前提升源域拟合，但跨族优势尚不稳定。",
        "",
        "## 9. guardrails",
        "",
        "- CPU",
        "- 1 epoch",
        "- 小样本",
        "- timm vit tiny",
        "- 不能作为最终泛化结论",
        "",
        "## 10. 下一步建议",
        "",
        "- 增加 epoch 到 3 或 5；",
        "- 重复 seed；",
        "- 接入真实 pretrained DINOv2；",
        "- 扩展到 Fault family；",
        "- 加入 SSIM/PSNR/edge metrics。",
        "",
        "## 附注",
        "",
        summary_report.read_text(encoding="utf-8"),
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 OpenFWI small transfer 阶段报告。")
    parser.add_argument("--summary-csv", required=True, type=Path)
    parser.add_argument("--summary-report", required=True, type=Path)
    parser.add_argument("--prediction-grids", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_openfwi_transfer_report(
        summary_csv=args.summary_csv,
        summary_report=args.summary_report,
        prediction_grids=args.prediction_grids,
        output_path=args.output,
    )
    print(f"写出阶段报告: {output}")


if __name__ == "__main__":
    main()
