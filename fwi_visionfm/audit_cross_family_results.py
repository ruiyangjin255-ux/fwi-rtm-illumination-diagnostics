from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fwi_visionfm.split_utils import load_split_paths, read_json


def _read_comparison(matrix_dir: Path) -> dict[str, Any]:
    path = matrix_dir / "comparison" / "comparison_summary.json"
    if not path.exists():
        return {"missing": str(path), "experiments": []}
    return read_json(path)


def _split_info(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    split_paths = load_split_paths(path)
    train_set = {str(item.resolve()) for item in split_paths["train"]}
    val_set = {str(item.resolve()) for item in split_paths["val"]}
    test_set = {str(item.resolve()) for item in split_paths["test"]}
    return {
        "path": str(path),
        "mode": payload.get("mode", ""),
        "protocol": payload.get("protocol", ""),
        "train_count": len(train_set),
        "val_count": len(val_set),
        "test_count": len(test_set),
        "train": train_set,
        "val": val_set,
        "test": test_set,
        "target_family": payload.get("target_family", ""),
        "source_families": payload.get("source_families", []),
        "families": payload.get("families", {}),
    }


def _checkpoint_exists(experiment_dir: Path) -> bool:
    if not experiment_dir.exists():
        return False
    return any(experiment_dir.rglob(pattern).__next__() for pattern in ("*.pt", "*.pth") if list(experiment_dir.rglob(pattern)))


def _safe_checkpoint_exists(experiment_dir: Path) -> bool:
    if not experiment_dir.exists():
        return False
    return any(experiment_dir.rglob("*.pt")) or any(experiment_dir.rglob("*.pth"))


def _format_split_table(split_infos: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| split_manifest | protocol | train | val | test | target_family | source_families |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for info in split_infos:
        lines.append(
            f"| {Path(info['path']).name} | {info['protocol'] or info['mode']} | {info['train_count']} | "
            f"{info['val_count']} | {info['test_count']} | {info['target_family']} | {','.join(info['source_families'])} |"
        )
    return lines


def audit_cross_family_results(
    *,
    experiment_dirs: list[str | Path],
    split_manifests: list[str | Path],
    output: str | Path,
) -> dict[str, Any]:
    matrix_records = []
    for matrix_raw in experiment_dirs:
        matrix_dir = Path(matrix_raw)
        comparison = _read_comparison(matrix_dir)
        matrix_records.append(
            {
                "name": matrix_dir.name,
                "path": str(matrix_dir),
                "comparison_count": len(comparison.get("experiments", [])),
                "has_comparison": "missing" not in comparison,
                "uses_final_val_metrics": any("final_val_mae" in row for row in comparison.get("experiments", [])),
                "checkpoint_exists": _safe_checkpoint_exists(matrix_dir),
            }
        )
    split_infos = [_split_info(Path(path)) for path in split_manifests]

    findings: list[str] = []
    if any(record["uses_final_val_metrics"] for record in matrix_records):
        findings.append("现有 comparison_summary 主要汇总 final_val_mae/final_val_rmse；这些不是严格的 held-out target-test 指标。")
    if any(info["val_count"] == 0 for info in split_infos):
        findings.append("部分 cross-family split 的 val_count=0，训练选择与评估记录不完整。")
    train_counts = {info["train_count"] for info in split_infos}
    if len(train_counts) > 1:
        findings.append("不同 split 的训练样本数不一致，不能直接作为公平泛化比较。")
    tests_by_target: dict[str, set[str]] = {}
    for info in split_infos:
        target = info["target_family"] or Path(info["path"]).stem
        if target in tests_by_target and tests_by_target[target] != info["test"]:
            findings.append(f"{target} 的 target test set 不一致；需要 matched target-test protocol。")
        tests_by_target[target] = info["test"]
    if not any(info["protocol"] == "protocol_v1_matched_target_test" for info in split_infos):
        findings.append("未检测到 Protocol v1 matched target-test split；旧结果只能作为工程 smoke 参考。")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Cross-Family Result Audit",
        "",
        "本审计用于识别旧 subset500 结果是否满足 matched target-test 公平比较要求。",
        "",
        "## Findings",
        "",
    ]
    if findings:
        lines.extend([f"- {item}" for item in findings])
    else:
        lines.append("- 未发现明显协议问题。")
    lines.extend(["", "## Split Manifests", "", *_format_split_table(split_infos), "", "## Matrix Artifacts", ""])
    lines.extend(
        [
            "| matrix | comparison_rows | has_comparison | checkpoint_exists |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for record in matrix_records:
        lines.append(
            f"| {record['name']} | {record['comparison_count']} | {record['has_comparison']} | {record['checkpoint_exists']} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            "使用 Protocol v1 matched target-test split 重新运行 in-domain 与 cross-family 矩阵；不要把旧 final_val_mae 当作目标 family test 指标。",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    result = {
        "output": str(output_path),
        "finding_count": len(findings),
        "findings": findings,
        "matrix_count": len(matrix_records),
        "split_count": len(split_infos),
    }
    (output_path.with_suffix(".json")).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 cross-family 实验结果是否满足 Protocol v1 公平比较要求。")
    parser.add_argument("--experiment-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--split-manifests", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_cross_family_results(
        experiment_dirs=args.experiment_dirs,
        split_manifests=args.split_manifests,
        output=args.output,
    )
    print(f"写出审计报告: {result['output']}")
    print(f"findings: {result['finding_count']}")


if __name__ == "__main__":
    main()
