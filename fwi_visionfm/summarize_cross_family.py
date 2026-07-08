from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def _extract_inline_value(text: str, field: str) -> str:
    match = re.search(rf"`{re.escape(field)}`:\s*`([^`]*)`", text)
    return match.group(1) if match else ""


def _parse_matrix_rows(text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in text.splitlines()]
    header_index = None
    for index, line in enumerate(lines):
        if line.startswith("| model_type |"):
            header_index = index
            break
    if header_index is None:
        return []
    rows: list[dict[str, str]] = []
    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 8:
            continue
        rows.append(
            {
                "model_type": parts[0],
                "final_val_mae": parts[1],
                "final_val_rmse": parts[2],
                "trainable_parameters": parts[3],
                "total_parameters": parts[4],
                "trainable_ratio": parts[5],
                "peft_type": parts[6],
                "injected_lora_modules": parts[7],
            }
        )
    return rows


def _load_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    text = report_path.read_text(encoding="utf-8")
    rows = _parse_matrix_rows(text)
    row_map = {row["model_type"]: row for row in rows}
    return {
        "path": str(report_path),
        "title": text.splitlines()[0].lstrip("# ").strip() if text.strip() else report_path.stem,
        "dataset_name": _extract_inline_value(text, "dataset_name"),
        "subset_name": _extract_inline_value(text, "subset_name"),
        "sample_count": _extract_inline_value(text, "sample_count"),
        "records_shape_set": _extract_inline_value(text, "records_shape_set"),
        "velocity_shape_set": _extract_inline_value(text, "velocity_shape_set"),
        "train_families": _extract_inline_value(text, "train_families"),
        "val_families": _extract_inline_value(text, "val_families"),
        "test_families": _extract_inline_value(text, "test_families"),
        "rows": rows,
        "row_map": row_map,
    }


def _safe_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _classify_report(report: dict[str, Any]) -> str:
    train_families = report["train_families"]
    test_families = report["test_families"]
    if train_families and test_families and train_families != test_families:
        return "cross_family"
    return "in_domain"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_model_label(record: dict[str, Any]) -> str:
    model_type = str(record.get("model_type", ""))
    peft_type = str(record.get("peft_type", "none"))
    if model_type == "frozen_foundation_baseline" and peft_type == "lora":
        return "dummy_dinov2_lora"
    if model_type == "frozen_foundation_baseline":
        return "dummy_dinov2_frozen"
    return model_type


def _split_families_from_manifest(path: Path) -> dict[str, str]:
    if not path.exists():
        return {"train": "", "val": "", "test": ""}
    payload = _read_json(path)
    family_meta = payload.get("families", {})
    result: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for split_name in ("train", "val", "test"):
        paths = [str(item).lower() for item in payload.get(split_name, [])]
        for family in family_meta:
            family_text = str(family).lower()
            if any(family_text in item for item in paths):
                result[split_name].append(str(family))
    for family in family_meta:
        if not any(family in values for values in result.values()):
            if len(family_meta) == 1:
                result["train"].append(str(family))
                result["val"].append(str(family))
                result["test"].append(str(family))
    return {key: ",".join(sorted(set(values))) for key, values in result.items()}


def _load_matrix_dir(matrix_dir: str | Path) -> dict[str, Any]:
    matrix_path = Path(matrix_dir)
    comparison_path = matrix_path / "comparison" / "comparison_summary.json"
    if not comparison_path.exists():
        raise FileNotFoundError(f"缺少 comparison_summary.json: {comparison_path}")
    comparison = _read_json(comparison_path)
    config_path = matrix_path / "matrix_config.json"
    run_summary_path = matrix_path / "matrix_run_summary.json"
    config = _read_json(config_path) if config_path.exists() else {}
    run_summary = _read_json(run_summary_path) if run_summary_path.exists() else {}
    split_manifest = Path(str(run_summary.get("split_manifest") or config.get("split_manifest") or ""))
    split_families = _split_families_from_manifest(split_manifest) if str(split_manifest) else {"train": "", "val": "", "test": ""}
    rows = []
    for record in comparison.get("experiments", []):
        rows.append(
            {
                "model_type": _normalize_model_label(record),
                "final_val_mae": str(record.get("final_val_mae", "")),
                "final_val_rmse": str(record.get("final_val_rmse", "")),
            }
        )
    return {
        "name": matrix_path.name,
        "path": str(matrix_path),
        "epochs": str(config.get("epochs", run_summary.get("epochs", ""))),
        "sample_count": "",
        "train_families": split_families["train"],
        "val_families": split_families["val"],
        "test_families": split_families["test"],
        "rows": rows,
        "row_map": {row["model_type"]: row for row in rows},
    }


def _is_cross_family_matrix(item: dict[str, Any]) -> bool:
    return bool(item["train_families"] and item["test_families"] and item["train_families"] != item["test_families"])


def summarize_cross_family_matrix_dirs(matrix_dirs: list[str | Path], output_path: str | Path) -> str:
    items = [_load_matrix_dir(path) for path in matrix_dirs if Path(path).exists()]
    in_domain = [item for item in items if not _is_cross_family_matrix(item)]
    cross = [item for item in items if _is_cross_family_matrix(item)]
    lines = [
        "# Cross-Family Subset500 Summary",
        "",
        "当前阶段是 OpenFWI first-file subset500 的 in-domain 与 cross-family 工程验证。",
        "subset500 和 dummy_dinov2 不能作为真实 Vision FM 泛化结论。",
        "",
        "## Experiments",
        "",
        "| experiment | train_family | val_family | test_family | epoch | model | final_val_mae | final_val_rmse |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in items:
        for row in item["rows"]:
            lines.append(
                f"| {item['name']} | {item['train_families']} | {item['val_families']} | {item['test_families']} | "
                f"{item['epochs']} | {row['model_type']} | {row['final_val_mae']} | {row['final_val_rmse']} |"
            )
    lines.extend(["", "## In-domain vs Cross-family Delta", ""])
    lines.append("| cross_experiment | model | in_domain_mae | cross_mae | mae_delta | in_domain_rmse | cross_rmse | rmse_delta |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    degradation_found = False
    for cross_item in cross:
        baseline = next((item for item in in_domain if item["test_families"] == cross_item["test_families"]), None)
        if baseline is None:
            continue
        for model in ("torch_cnn_baseline", "dummy_dinov2_frozen", "dummy_dinov2_lora"):
            base_row = baseline["row_map"].get(model, {})
            cross_row = cross_item["row_map"].get(model, {})
            base_mae = _safe_float(base_row.get("final_val_mae", ""))
            cross_mae = _safe_float(cross_row.get("final_val_mae", ""))
            base_rmse = _safe_float(base_row.get("final_val_rmse", ""))
            cross_rmse = _safe_float(cross_row.get("final_val_rmse", ""))
            mae_delta = "" if base_mae is None or cross_mae is None else f"{cross_mae - base_mae:.3f}"
            rmse_delta = "" if base_rmse is None or cross_rmse is None else f"{cross_rmse - base_rmse:.3f}"
            if base_mae is not None and cross_mae is not None and cross_mae > base_mae:
                degradation_found = True
            lines.append(
                f"| {cross_item['name']} | {model} | {'' if base_mae is None else base_mae} | "
                f"{'' if cross_mae is None else cross_mae} | {mae_delta} | "
                f"{'' if base_rmse is None else base_rmse} | {'' if cross_rmse is None else cross_rmse} | {rmse_delta} |"
            )
    lines.extend(
        [
            "",
            "## Conclusions",
            "",
            "- " + ("cross-family 相比对应 in-domain 出现泛化下降。" if degradation_found else "当前未观察到明确泛化下降。"),
            "- 当前不能声称 dummy_dinov2_frozen 或 dummy_dinov2_lora 相对 CNN 有稳定优势，除非跨 family 指标持续支持。",
            "- dummy_dinov2 不是真实 DINOv2 预训练模型，不能作为最终 Vision FM 结论。",
        ]
    )
    content = "\n".join(lines)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return content


def summarize_cross_family_reports(report_paths: list[str | Path], output_path: str | Path) -> str:
    reports = [_load_report(path) for path in report_paths]
    in_domain_reports = [report for report in reports if _classify_report(report) == "in_domain"]
    cross_family_reports = [report for report in reports if _classify_report(report) == "cross_family"]

    lines = [
        "# Cross-Family Summary",
        "",
        "当前目标是从 FlatVel-A in-domain baseline 推进到 CurveVel/Fault cross-family 泛化验证。",
        "只有在 cross-family 中出现稳定优势，才有资格讨论 foundation-style 表征是否改善泛化。",
        "当前 dummy_dinov2 和 dummy_lora 仍然只能说明接口与工程闭环可用，不能代表真实 Vision FM 迁移结论。",
        "",
        "## Experiments",
        "",
        "| report | train_families | val_families | test_families | sample_count | records_shape_set | velocity_shape_set |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for report in reports:
        lines.append(
            f"| {report['subset_name'] or report['title']} | {report['train_families']} | {report['val_families']} | "
            f"{report['test_families']} | {report['sample_count']} | {report['records_shape_set']} | {report['velocity_shape_set']} |"
        )

    lines.extend(["", "## Model Metrics", ""])
    for report in reports:
        lines.append(f"### {report['subset_name'] or report['title']}")
        lines.append("")
        lines.append("| model_type | final_val_mae | final_val_rmse |")
        lines.append("| --- | --- | --- |")
        for row in report["rows"]:
            lines.append(f"| {row['model_type']} | {row['final_val_mae']} | {row['final_val_rmse']} |")
        lines.append("")

    lines.extend(["## In-domain vs Cross-family", ""])
    degradation_found = False
    for cross_report in cross_family_reports:
        matched_baseline = None
        for candidate in in_domain_reports:
            if candidate["test_families"] == cross_report["test_families"]:
                matched_baseline = candidate
                break
        if matched_baseline is None:
            continue
        lines.append(f"### {cross_report['subset_name'] or cross_report['title']}")
        lines.append("")
        lines.append("| model_type | in_domain_mae | cross_family_mae | mae_delta | in_domain_rmse | cross_family_rmse | rmse_delta |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for model_type in ("torch_cnn_baseline", "frozen_foundation_baseline", "foundation+lora"):
            in_row = matched_baseline["row_map"].get(model_type, {})
            cross_row = cross_report["row_map"].get(model_type, {})
            in_mae = _safe_float(in_row.get("final_val_mae"))
            cross_mae = _safe_float(cross_row.get("final_val_mae"))
            in_rmse = _safe_float(in_row.get("final_val_rmse"))
            cross_rmse = _safe_float(cross_row.get("final_val_rmse"))
            mae_delta = "" if in_mae is None or cross_mae is None else f"{cross_mae - in_mae:.3f}"
            rmse_delta = "" if in_rmse is None or cross_rmse is None else f"{cross_rmse - in_rmse:.3f}"
            if in_mae is not None and cross_mae is not None and cross_mae > in_mae:
                degradation_found = True
            lines.append(
                f"| {model_type} | {'' if in_mae is None else in_mae} | {'' if cross_mae is None else cross_mae} | {mae_delta} | "
                f"{'' if in_rmse is None else in_rmse} | {'' if cross_rmse is None else cross_rmse} | {rmse_delta} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Conclusions",
            "",
            "- " + ("出现明显泛化下降。" if degradation_found else "当前未观察到明确的跨 family 泛化下降信号。"),
            "- 当前没有证据表明 dummy foundation / dummy LoRA 相对 CNN 具有稳定优势。",
            "- dummy_dinov2 不是真实 DINOv2 预训练结论。",
        ]
    )

    content = "\n".join(lines)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return content


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 in-domain 与 cross-family 报告。")
    parser.add_argument("--reports", nargs="+", type=Path)
    parser.add_argument("--matrix-dirs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.matrix_dirs:
        content = summarize_cross_family_matrix_dirs(args.matrix_dirs, args.output)
    elif args.reports:
        content = summarize_cross_family_reports(args.reports, args.output)
    else:
        raise SystemExit("错误: --reports 或 --matrix-dirs 至少提供一个。")
    print(f"写出汇总: {args.output}")
    print(f"汇总长度: {len(content.splitlines())} 行")


if __name__ == "__main__":
    main()
