from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _experiment_dirs(root_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in root_dir.iterdir()
        if path.is_dir() and (path / "config_resolved.json").exists()
    )


def _load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _to_float(value: Any, default: str | float = "NA") -> Any:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def collect_real_dinov2_test_smoke(
    *,
    root_dir: str | Path,
    output_dir: str | Path | None = None,
    evaluation_tag: str = "smoke",
) -> dict[str, Any]:
    root_dir = Path(root_dir)
    output_dir = root_dir if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_tag = str(evaluation_tag)
    if evaluation_tag not in {"smoke", "fulltest"}:
        raise ValueError(f"Unsupported evaluation_tag: {evaluation_tag}")

    rows: list[dict[str, Any]] = []
    completed_in_family = True
    completed_cross_family = True
    checkpoint_load_failure = False
    image_size_restore_failure = False
    backbone_config_restore_failure = False
    cpu_timeout_occurred = False
    ssim_unavailable = False

    for experiment_dir in _experiment_dirs(root_dir):
        config = _read_json(experiment_dir / "config_resolved.json")
        summary = _load_metrics(experiment_dir / "foundation_experiment_summary.json")
        in_metrics = _load_metrics(experiment_dir / f"metrics_in_family_{evaluation_tag}.json")
        cross_metrics = _load_metrics(experiment_dir / f"metrics_cross_family_{evaluation_tag}.json")

        if not in_metrics:
            completed_in_family = False
        if not cross_metrics:
            completed_cross_family = False
        if summary == {}:
            checkpoint_load_failure = True

        image_size = config.get("image_size", summary.get("image_size", "NA"))
        backbone_type = summary.get("backbone_type", config.get("backbone_type", "NA"))
        backbone_name = summary.get("backbone_name", config.get("model_name", "NA"))
        if backbone_type == "NA" or backbone_name == "NA":
            backbone_config_restore_failure = True
        if str(summary.get("backbone_name", config.get("model_name", ""))) == "vit_small_patch14_dinov2.lvd142m" and int(image_size) != 518:
            image_size_restore_failure = True

        if in_metrics.get("ssim_available") is False or cross_metrics.get("ssim_available") is False:
            ssim_unavailable = True

        in_mae = _to_float(in_metrics.get("mae"))
        in_rmse = _to_float(in_metrics.get("rmse"))
        cross_mae = _to_float(cross_metrics.get("mae"))
        cross_rmse = _to_float(cross_metrics.get("rmse"))
        gap_mae = "NA" if "NA" in (in_mae, cross_mae) else float(cross_mae) - float(in_mae)
        gap_rmse = "NA" if "NA" in (in_rmse, cross_rmse) else float(cross_rmse) - float(in_rmse)

        eval_samples = in_metrics.get("sample_count", cross_metrics.get("sample_count", "NA"))
        rows.append(
            {
                "experiment": experiment_dir.name,
                "transfer_mode": summary.get("transfer_mode", config.get("transfer_mode", "NA")),
                "peft": summary.get("peft_type", config.get("peft_type", "NA")),
                "backbone_type": backbone_type,
                "backbone_name": backbone_name,
                "image_size": image_size,
                "eval_samples": eval_samples,
                "eval_max_samples": in_metrics.get("eval_max_samples", cross_metrics.get("eval_max_samples", 32)),
                "in_family_mae": in_mae,
                "in_family_rmse": in_rmse,
                "in_family_psnr": _to_float(in_metrics.get("psnr")),
                "in_family_edge_mae": _to_float(in_metrics.get("edge_mae")),
                "in_family_laplacian_mae": _to_float(in_metrics.get("laplacian_mae")),
                "cross_family_mae": cross_mae,
                "cross_family_rmse": cross_rmse,
                "cross_family_psnr": _to_float(cross_metrics.get("psnr")),
                "cross_family_edge_mae": _to_float(cross_metrics.get("edge_mae")),
                "cross_family_laplacian_mae": _to_float(cross_metrics.get("laplacian_mae")),
                "cross_family_laplian_mae": _to_float(cross_metrics.get("laplacian_mae")),
                "generalization_gap_mae": gap_mae,
                "generalization_gap_rmse": gap_rmse,
                "trainable_params": summary.get("trainable_parameters", "NA"),
                "trainable_ratio": summary.get("trainable_ratio", "NA"),
                "output_dir": str(experiment_dir),
            }
        )

    if not rows:
        raise ValueError(f"No real DINOv2 smoke experiments found under {root_dir}")

    if evaluation_tag == "smoke":
        csv_path = output_dir / "real_dinov2_test_smoke_summary.csv"
        md_path = output_dir / "real_dinov2_test_smoke_summary.md"
        report_path = output_dir / "real_dinov2_test_smoke_report.md"
        title = "Real DINOv2 Test Smoke"
        scope_lines = [
            "- 本轮只做 checkpoint-only smoke evaluation，不重新训练。",
            "- 本轮结果只覆盖 `max_samples=32` 的固定小子集。",
            "- 这不是正式 DINOv2 cross-family 结论。",
            "- 下一步应扩大评估样本数，或转到 GPU 进行更稳定的真实 DINOv2 评估。",
        ]
    else:
        csv_path = output_dir / "real_dinov2_fulltest_summary.csv"
        md_path = output_dir / "real_dinov2_fulltest_summary.md"
        report_path = output_dir / "real_dinov2_fulltest_report.md"
        title = "Real DINOv2 Full Fixed-Test"
        scope_lines = [
            "- 这是 checkpoint-only full fixed-test evaluation，不重新训练。",
            "- 当前 checkpoint 仍只来自 `smoke_train=32` 训练，因此不能作为正式 DINOv2 性能。",
            "- eval split 为完整 `100 / 100` in-family / cross-family fixed test。",
            "- 如果要正式 DINOv2 结论，应转 GPU 并用完整 train split 重新训练。",
        ]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        f"# {title} Summary",
        "",
        "| experiment | transfer_mode | in_family_mae | in_family_rmse | cross_family_mae | cross_family_rmse | generalization_gap_mae | trainable_ratio |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['experiment']} | {row['transfer_mode']} | {row['in_family_mae']} | {row['in_family_rmse']} | "
            f"{row['cross_family_mae']} | {row['cross_family_rmse']} | {row['generalization_gap_mae']} | {row['trainable_ratio']} |"
        )
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    lora_better = False
    if len(rows) == 2:
        frozen_row = next((row for row in rows if row["transfer_mode"] == "frozen"), None)
        lora_row = next((row for row in rows if row["transfer_mode"] == "lora"), None)
        if frozen_row and lora_row and frozen_row["cross_family_mae"] != "NA" and lora_row["cross_family_mae"] != "NA":
            lora_better = float(lora_row["cross_family_mae"]) < float(frozen_row["cross_family_mae"])

    report_lines = [
        f"# {title} Report",
        "",
        "## Status",
        "",
        f"- Frozen / LoRA DINOv2 checkpoints complete test_in_family {evaluation_tag}: `{completed_in_family}`",
        f"- Frozen / LoRA DINOv2 checkpoints complete test_cross_family {evaluation_tag}: `{completed_cross_family}`",
        f"- CPU timeout occurred: `{cpu_timeout_occurred}`",
        f"- checkpoint load failure occurred: `{checkpoint_load_failure}`",
        f"- image_size config restore failure occurred: `{image_size_restore_failure}`",
        f"- backbone config restore failure occurred: `{backbone_config_restore_failure}`",
        "",
        "## Scope",
        "",
        *scope_lines,
        "",
        "## Metrics",
        "",
        *md_lines[2:],
        "",
        "## Interpretation",
        "",
        f"- frozen / LoRA 是否都成功: `{completed_in_family and completed_cross_family and not checkpoint_load_failure and not image_size_restore_failure and not backbone_config_restore_failure}`",
        f"- LoRA 是否仍优于 frozen: `{lora_better}`",
        "",
        "## Notes",
        "",
        "- ssim unavailable does not block the smoke report.",
        "- 如果后续出现 checkpoint 缺失、超时或 image_size 恢复失败，应优先修复评估兼容性，而不是解读数值。",
    ]
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "row_count": len(rows),
        "csv_path": str(csv_path),
        "md_path": str(md_path),
        "report_path": str(report_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 real DINOv2 checkpoint-only test smoke 结果。")
    parser.add_argument("--root", "--root-dir", dest="root_dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--evaluation-tag", choices=("smoke", "fulltest"), default="smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = collect_real_dinov2_test_smoke(root_dir=args.root_dir, output_dir=args.output_dir, evaluation_tag=args.evaluation_tag)
    print(f"写出汇总 CSV: {result['csv_path']}")
    print(f"写出汇总报告: {result['report_path']}")


if __name__ == "__main__":
    main()
