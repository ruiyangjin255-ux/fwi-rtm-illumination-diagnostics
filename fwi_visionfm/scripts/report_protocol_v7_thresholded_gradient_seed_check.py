from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _load_rows(root: Path) -> list[dict[str, str]]:
    with (root / "protocol_v7_thresholded_gradient_seed_check_summary.csv").open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rows_by_method(rows: list[dict[str, str]], method: str) -> list[dict[str, str]]:
    return [row for row in rows if row["boundary_method"] == method and row["status"] == "SUCCESS"]


def _wins(thresholded: list[dict[str, str]], baseline: list[dict[str, str]]) -> tuple[int, int]:
    thresholded_by_seed = {row["seed"]: row for row in thresholded}
    baseline_by_seed = {row["seed"]: row for row in baseline}
    grad_wins = 0
    edge_wins = 0
    for seed, row in thresholded_by_seed.items():
        other = baseline_by_seed.get(seed)
        if not other:
            continue
        if float(row["gradient_error"]) < float(other["gradient_error"]):
            grad_wins += 1
        if float(row["edge_MAE"]) < float(other["edge_MAE"]):
            edge_wins += 1
    return grad_wins, edge_wins


def write_protocol_v7_thresholded_gradient_seed_check_report(
    root: str | Path,
    reuse_tuning_root: str | Path | None = None,
    reuse_seed_stability_root: str | Path | None = None,
) -> Path:
    del reuse_tuning_root, reuse_seed_stability_root
    root_path = Path(root)
    rows = _load_rows(root_path)
    thresholded = _rows_by_method(rows, "thresholded_gradient")
    gm005 = _rows_by_method(rows, "gradient_magnitude_lambda005")
    gm010 = _rows_by_method(rows, "gradient_magnitude_lambda010")
    grad_wins_005, edge_wins_005 = _wins(thresholded, gm005)
    grad_wins_010, edge_wins_010 = _wins(thresholded, gm010)

    lines = [
        "# Protocol V7 Thresholded Gradient Seed Check Report",
        "",
        "## 1. Goal",
        "本轮只补 thresholded_gradient + lambda=0.05 的 seed=1/2，验证该 method 的结构收益是否能从 seed=0 扩展到多 seed。",
        "",
        "## 2. Reused Results",
        "- reused_from 字段保留自 tuning summary。",
        "",
        "## 3. Three-way Comparison",
        "| seed | method | lambda_boundary | threshold | MAE | RMSE | SSIM | gradient_error | edge_MAE | reused_from |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['seed']} | {row['boundary_method']} | {row['lambda_boundary']} | {row['threshold']} | {row['MAE']} | {row['RMSE']} | {row['SSIM']} | {row['gradient_error']} | {row['edge_MAE']} | {row['reused_from']} |"
        )
    lines.extend(
        [
            "",
            "## 4. Interpretation",
            f"- thresholded_gradient vs gradient_magnitude + lambda=0.05: gradient_error wins {grad_wins_005}/3, edge_MAE wins {edge_wins_005}/3.",
            f"- thresholded_gradient vs current recommended gradient_magnitude + lambda=0.10: gradient_error wins {grad_wins_010}/3, edge_MAE wins {edge_wins_010}/3.",
            "- 需要同时检查 MAE/RMSE 是否出现明显代价，再决定是否升级推荐。",
            "",
            "## 5. Recommendation Check",
        ]
    )
    if grad_wins_010 >= 2 and edge_wins_010 >= 2:
        lines.append("- thresholded_gradient 达到至少 2/3 seed 结构优势，值得考虑替代当前推荐，但仍需保持 selected-seed 口径。")
    else:
        lines.append("- 若 thresholded_gradient 只在 seed=0 优势明显或未达到 2/3 结构优势，则不升级当前推荐。")
    lines.extend(
        [
            "",
            "## 6. Limitations",
            "- CPU small-sample",
            "- only seed=0/1/2",
            "- selected method check only",
            "- not benchmark-level proof",
            "- no DINOv2/SAM/NCS",
            "",
            "## 7. Guardrail",
            "- 本报告不写泛化提升结论。",
        ]
    )
    report_path = root_path / "protocol_v7_thresholded_gradient_seed_check_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Protocol V7 thresholded-gradient seed check report.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reuse-tuning-root", type=Path, default=None)
    parser.add_argument("--reuse-seed-stability-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        write_protocol_v7_thresholded_gradient_seed_check_report(
            args.root,
            reuse_tuning_root=args.reuse_tuning_root,
            reuse_seed_stability_root=args.reuse_seed_stability_root,
        )
    )


if __name__ == "__main__":
    main()
