from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fwi_visionfm.scripts.run_protocol_v7_boundary_auxiliary_seed_stability import (
    build_seed_stability_summary,
    compute_seed_stability_win_counts,
)


def _pair_rows(rows: list[dict[str, str]]) -> list[tuple[str, dict[str, str] | None, dict[str, str] | None]]:
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["seed"], {})[row["model_type"]] = row
    ordered = []
    for seed in sorted(grouped.keys(), key=int):
        seed_rows = grouped[seed]
        ordered.append((seed, seed_rows.get("baseline"), seed_rows.get("boundary_aux")))
    return ordered


def _to_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _majority_text(wins: dict[str, int]) -> tuple[str, str]:
    structural = wins["gradient_error_lower"] >= 2 and wins["edge_MAE_lower"] >= 2
    numerical_tradeoff = wins["MAE_lower"] < 2 or wins["RMSE_lower"] < 2
    structural_text = "selected boundary auxiliary shows majority-supported structural benefit" if structural else "selected boundary auxiliary does not show majority-supported structural benefit"
    tradeoff_text = "numerical trade-off exists" if numerical_tradeoff else "numerical trade-off does not exist"
    return structural_text, tradeoff_text


def write_seed_stability_report(root: str | Path, reuse_seed0_root: str | Path | None = None) -> Path:
    output_root = Path(root)
    summary_path = output_root / "protocol_v7_boundary_auxiliary_seed_stability_summary.csv"
    if not summary_path.exists():
        summary_path = build_seed_stability_summary(root=output_root, reuse_seed0_root=reuse_seed0_root)
    rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8")))
    wins = compute_seed_stability_win_counts(rows)
    structural_text, tradeoff_text = _majority_text(wins)
    lines = [
        "# Protocol V7 Boundary Auxiliary Selected Seed Stability Report",
        "",
        "## 1. Goal",
        "本轮只验证 selected boundary auxiliary 配置在 seed=0/1/2 下是否具有稳定结构收益，不做 benchmark claim。",
        "",
        "## 2. Matched Settings",
        "- baseline 与 boundary_aux 仅在 decoder/loss 及 boundary auxiliary 参数上不同。",
        "- 其余设置保持对齐：cnn_baseline、raw_envelope_spectrum3、geometry disabled、mean aggregator、CPU small-sample、physical_velocity metric space。",
        "",
        "## 3. Seed Stability Table",
        "| seed | baseline_status | boundary_status | baseline_MAE | boundary_MAE | baseline_RMSE | boundary_RMSE | baseline_SSIM | boundary_SSIM | baseline_gradient_error | boundary_gradient_error | baseline_edge_MAE | boundary_edge_MAE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for seed, baseline, boundary in _pair_rows(rows):
        lines.append(
            f"| {seed} | {(baseline or {}).get('status', 'SKIPPED')} | {(boundary or {}).get('status', 'SKIPPED')} | "
            f"{(baseline or {}).get('cross_family_MAE', '')} | {(boundary or {}).get('cross_family_MAE', '')} | "
            f"{(baseline or {}).get('cross_family_RMSE', '')} | {(boundary or {}).get('cross_family_RMSE', '')} | "
            f"{(baseline or {}).get('cross_family_SSIM', '')} | {(boundary or {}).get('cross_family_SSIM', '')} | "
            f"{(baseline or {}).get('cross_family_gradient_error', '')} | {(boundary or {}).get('cross_family_gradient_error', '')} | "
            f"{(baseline or {}).get('cross_family_edge_MAE', '')} | {(boundary or {}).get('cross_family_edge_MAE', '')} |"
        )
    lines.extend(
        [
            "",
            "## 4. Win Counts",
            f"- MAE lower: {wins['MAE_lower']}",
            f"- RMSE lower: {wins['RMSE_lower']}",
            f"- SSIM higher: {wins['SSIM_higher']}",
            f"- gradient_error lower: {wins['gradient_error_lower']}",
            f"- edge_MAE lower: {wins['edge_MAE_lower']}",
            "",
            "## 5. Interpretation",
            f"- {structural_text}.",
            f"- {tradeoff_text}.",
            "- results remain CPU small-sample selected-seed evidence.",
            "",
            "## 6. Limitations",
            "- CPU small-sample",
            "- smoke-scale selected setting",
            "- only seed=0/1/2",
            "- boundary target is gradient-derived, not manually interpreted geology",
            "- not benchmark-level proof",
            "- no DINOv2/SAM/NCS in this run",
            "",
            "## 7. Next Step",
        ]
    )
    if wins["gradient_error_lower"] >= 2 and wins["edge_MAE_lower"] >= 2 and wins["MAE_lower"] >= 2 and wins["RMSE_lower"] >= 2:
        lines.append("- 结构指标在至少 2/3 seeds 胜出且 MAE/RMSE 未明显恶化，可进入 V7 selected multi-seed report。")
    elif wins["gradient_error_lower"] < 2 or wins["edge_MAE_lower"] < 2:
        lines.append("- 结构指标稳定性不足，不扩大矩阵，优先调 lambda_boundary=0.01/0.03 或改 boundary target。")
    else:
        lines.append("- MAE/RMSE 代价偏明显，优先降低 lambda_boundary 或改为 weak boundary auxiliary loss。")
    failed_or_skipped = [row for row in rows if row["status"] in {"FAILED", "SKIPPED"}]
    if failed_or_skipped:
        lines.extend(["", "### Missing or Failed Runs"])
        for row in failed_or_skipped:
            lines.append(f"- seed={row['seed']} model_type={row['model_type']} status={row['status']} reason={row['skip_reason']}")
    report_path = output_root / "protocol_v7_boundary_auxiliary_seed_stability_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Protocol V7 boundary auxiliary seed stability report.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reuse-seed0-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(write_seed_stability_report(args.root, reuse_seed0_root=args.reuse_seed0_root))


if __name__ == "__main__":
    main()
