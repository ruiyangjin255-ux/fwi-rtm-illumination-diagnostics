from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_stage_report(
    *,
    protocol_summary: str | Path,
    results_index: str | Path,
    eval_csv: str | Path,
    figures_dir: str | Path,
    output: str | Path,
) -> str:
    protocol_summary_path = Path(protocol_summary)
    results_index_path = Path(results_index)
    eval_csv_path = Path(eval_csv)
    figures_dir = Path(figures_dir)
    output_path = Path(output)

    index_payload = _read_json(results_index_path)
    eval_rows = _read_csv(eval_csv_path)
    complete_rows = [row for row in eval_rows if row.get("status") == "complete"]
    best_row = min(complete_rows, key=lambda row: float(row["mae"])) if complete_rows else None

    lines = [
        "# FWI-VisionFM CPU Protocol v1 Stage Report",
        "",
        "## 1. 当前阶段目标",
        "",
        "当前阶段已经停止大规模训练。",
        "后续工作以程序框架、评估可视化和可复现实验管线为主。",
        "",
        "## 2. 数据与协议",
        "",
        f"- `protocol_summary`: `{protocol_summary_path}`",
        "- Protocol v1 使用 matched target-test split。",
        "- subset500 + CPU + 3 epoch 是小规模工程验证。",
        "",
        "## 3. 已完成程序模块",
        "",
        "- result indexing",
        "- existing checkpoint evaluation",
        "- prediction/error visualization",
        "- protocol comparison plotting",
        "- stage report generation",
        "- config-based pipeline preparation",
        "",
        "## 4. 已完成小规模训练矩阵",
        "",
        f"- `indexed_experiments`: `{len(index_payload.get('experiments', []))}`",
        "",
        "## 5. 测试集指标汇总",
        "",
    ]
    if best_row is not None:
        lines.extend(
            [
                f"- `best_experiment`: `{best_row['experiment']}`",
                f"- `best_model`: `{best_row['model_name']}`",
                f"- `best_test_mae`: `{best_row['mae']}`",
                f"- `best_test_rmse`: `{best_row['rmse']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Metric Source",
            "",
            f"- `test metrics`: `{eval_csv_path}`",
            f"- `protocol summary`: `{protocol_summary_path}`",
            "- `best model criterion`: minimum `mae` from `all_test_metrics.csv` rows with `status=complete`",
            "- `best_test_rmse`: taken from the same best row in `all_test_metrics.csv`, not from `final_val_rmse`",
            "",
            "## 6. In-domain vs Cross-family 对比",
            "",
            "当前以 Protocol v1 test metrics 为准，不再把 final_val 指标误写为 test 指标。",
            "",
            "## 7. CNN vs dummy foundation vs dummy LoRA 对比",
            "",
            "dummy_dinov2 只是工程接口，不是实时下载的真实预训练 Vision FM。",
            "",
            "## 8. 预测图与误差图观察",
            "",
            f"- `figures_dir`: `{figures_dir}`",
            "",
            "## 9. 当前结论边界",
            "",
            "当前不做大规模训练。",
            "当前结果不代表真实 Vision FM 结论。",
            "dummy_dinov2 只是工程接口。",
            "任何 dummy_dinov2 结果都不能表述为真实 Vision Foundation Model 泛化结论。",
            "",
            "## 10. 下一步程序构建计划",
            "",
            "- 保持 CPU-only 后处理管线可复现。",
            "- 真实 DINOv2 仅做 interface smoke。",
            "- 等 GPU 后再进入真实 backbone 训练。",
        ]
    )
    report = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 CPU Protocol v1 阶段性研究报告。")
    parser.add_argument("--protocol-summary", required=True, type=Path)
    parser.add_argument("--results-index", required=True, type=Path)
    parser.add_argument("--eval-csv", required=True, type=Path)
    parser.add_argument("--figures-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_stage_report(
        protocol_summary=args.protocol_summary,
        results_index=args.results_index,
        eval_csv=args.eval_csv,
        figures_dir=args.figures_dir,
        output=args.output,
    )
    print(f"写出阶段报告: {args.output}")
    print(f"报告长度: {len(report.splitlines())} 行")


if __name__ == "__main__":
    main()
