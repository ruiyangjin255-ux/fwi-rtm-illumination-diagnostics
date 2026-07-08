from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from fwi_visionfm.split_utils import read_json


MODEL_LABELS = {
    "torch_cnn_baseline": "torch_cnn_baseline",
    "frozen_foundation_baseline": "dummy_dinov2_frozen",
    "foundation+lora": "dummy_dinov2_lora",
}


def _read_comparison_rows(matrix_dir: Path) -> list[dict[str, Any]]:
    json_path = matrix_dir / "comparison" / "comparison_summary.json"
    csv_path = matrix_dir / "comparison" / "comparison_summary.csv"
    if json_path.exists():
        payload = read_json(json_path)
        return list(payload.get("experiments", []))
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    return []


def _normalize_model(record: dict[str, Any]) -> str:
    model_type = str(record.get("model_type", ""))
    peft_type = str(record.get("peft_type", "none"))
    if model_type == "frozen_foundation_baseline" and peft_type == "lora":
        return "dummy_dinov2_lora"
    return MODEL_LABELS.get(model_type, model_type)


def _read_test_metrics(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("test_mae", "") != "" or record.get("test_rmse", "") != "":
        return {
            "test_mae": record.get("test_mae", ""),
            "test_rmse": record.get("test_rmse", ""),
        }
    experiment_dir_raw = record.get("experiment_dir", "")
    if not experiment_dir_raw:
        return {"test_mae": "", "test_rmse": ""}
    experiment_dir = Path(str(experiment_dir_raw))
    for summary_name in ("torch_experiment_summary.json", "foundation_experiment_summary.json"):
        summary_path = experiment_dir / summary_name
        if summary_path.exists():
            summary = read_json(summary_path)
            test_metrics = summary.get("test_metrics", {})
            return {
                "test_mae": test_metrics.get("mae", ""),
                "test_rmse": test_metrics.get("rmse", ""),
            }
    return {"test_mae": "", "test_rmse": ""}


def _split_counts(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    return {
        "name": path.stem,
        "path": str(path),
        "train_count": len(payload.get("train", [])),
        "val_count": len(payload.get("val", [])),
        "test_count": len(payload.get("test", [])),
        "target_family": payload.get("target_family", ""),
        "source_families": ",".join(payload.get("source_families", [])),
        "protocol": payload.get("protocol", ""),
    }


def summarize_protocol_v1(
    *,
    matrix_root: str | Path,
    split_dir: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    matrix_root = Path(matrix_root)
    split_dir = Path(split_dir)
    split_paths = sorted(path for path in split_dir.glob("protocol_v1_*.json") if path.name != "protocol_v1_summary.json")
    split_by_name = {path.stem: _split_counts(path) for path in split_paths}
    rows: list[dict[str, Any]] = []
    for split_name, split_info in split_by_name.items():
        matrix_dir = matrix_root / split_name
        for record in _read_comparison_rows(matrix_dir):
            test_metrics = _read_test_metrics(record)
            rows.append(
                {
                    "experiment": split_name,
                    "target_family": split_info["target_family"],
                    "source_families": split_info["source_families"],
                    "train_count": split_info["train_count"],
                    "val_count": split_info["val_count"],
                    "test_count": split_info["test_count"],
                    "model": _normalize_model(record),
                    "final_val_mae": record.get("final_val_mae", ""),
                    "final_val_rmse": record.get("final_val_rmse", ""),
                    "test_mae": test_metrics["test_mae"],
                    "test_rmse": test_metrics["test_rmse"],
                    "trainable_parameters": record.get("trainable_parameters", ""),
                    "total_parameters": record.get("total_parameters", ""),
                    "trainable_ratio": record.get("trainable_ratio", ""),
                    "peft_type": record.get("peft_type", ""),
                    "injected_lora_modules": record.get("injected_lora_modules", ""),
                }
            )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Protocol v1: Matched Target-Test Cross-Family Summary",
        "",
        "Protocol v1 固定每个 target family 的 test set，并匹配 in-domain 与 cross-family 的训练样本数。",
        "当前矩阵输出仍以训练脚本中的 `final_val_mae` / `final_val_rmse` 为主；这些字段用于工程对比，不应被表述为最终 test metric。",
        "",
        "## Splits",
        "",
        "| split | source_families | target_family | train | val | test | protocol |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for split_info in split_by_name.values():
        lines.append(
            f"| {split_info['name']} | {split_info['source_families']} | {split_info['target_family']} | "
            f"{split_info['train_count']} | {split_info['val_count']} | {split_info['test_count']} | {split_info['protocol']} |"
        )
    lines.extend(
        [
            "",
            "## Matrix Metrics",
            "",
            "| experiment | source_families | target_family | model | final_val_mae | final_val_rmse | test_mae | test_rmse | trainable_ratio | peft_type | injected_lora_modules |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['source_families']} | {row['target_family']} | {row['model']} | "
            f"{row['final_val_mae']} | {row['final_val_rmse']} | {row['test_mae']} | {row['test_rmse']} | "
            f"{row['trainable_ratio']} | {row['peft_type']} | {row['injected_lora_modules']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Protocol v1 用于修正旧 subset500 比较中的 target-test 不匹配问题。",
            "- dummy_dinov2 不是真实 DINOv2 预训练权重，不能作为真实 foundation model 泛化结论。",
            "- subset500 和 3 epoch CPU 结果仍是小规模工程验证，不作为最终科研性能结论。",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "output": str(output_path),
        "split_count": len(split_by_name),
        "matrix_count": len({row["experiment"] for row in rows}),
        "row_count": len(rows),
        "rows": rows,
    }
    output_path.with_suffix(".json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 Protocol v1 matched target-test 实验矩阵。")
    parser.add_argument("--matrix-root", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = summarize_protocol_v1(matrix_root=args.matrix_root, split_dir=args.split_dir, output=args.output)
    print(f"写出 Protocol v1 汇总: {payload['output']}")
    print(f"matrix_count: {payload['matrix_count']}")


if __name__ == "__main__":
    main()
