from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fwi_visionfm.scripts.run_protocol_v6_geometry_aggregation_smoke import summarize_protocol_v6_runs


def write_protocol_v6_geometry_aggregation_report(root: str | Path) -> Path:
    output_root = Path(root)
    summary_path = output_root / "protocol_v6_geometry_aggregation_summary.csv"
    if not summary_path.exists():
        summary_path = summarize_protocol_v6_runs(output_root)
    rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8")))
    lines = [
        "# Protocol V6 Geometry-aware Bridge and Source-aware Aggregation Smoke Report",
        "",
        "## 1. Goal",
        "本轮只验证 geometry-aware bridge 和 source-aware aggregation 是否能进入真实训练与评价链路，不做 benchmark claim。",
        "",
        "## 2. Runs",
    ]
    for row in rows:
        lines.append(
            f"- {row['run_id']}: bridge={row['bridge']}, geometry_enabled={row['geometry_enabled']}, aggregator={row['aggregator']}, status={row['status']}"
        )
    lines.extend(
        [
            "",
            "## 3. Geometry-aware Bridge Check",
            "- geometry.enabled=false 用于确认旧 baseline 仍可运行。",
            "- geometry.enabled=true 的 run 若成功，说明 geometry embedding、geometry_config 保存与 projection_to_3ch 路径可进入真实训练。",
            "- projection_to_3ch 的存在保证了 CNN 以及 future VisionFM 的固定 3-channel 输入兼容性。",
            "",
            "## 4. Aggregator Check",
            "- mean / attention / source_aware_attention 都只做 smoke-scale chain validation。",
            "- 若 attention 类 run 成功并落盘 aggregator_attention_weights.npz，则说明 attention_weights 保存路径可用。",
            "- attention entropy 来自保存的 attention_weights，用于后续可解释性分析。",
            "- source-aware attention 在缺少真实几何 metadata 时，当前只使用 fallback index/source position 信息。",
            "",
            "## 5. Metrics",
        ]
    )
    for row in rows:
        lines.append(
            f"- {row['run_id']}: val(MAE={row['val_MAE']}, RMSE={row['val_RMSE']}, SSIM={row['val_SSIM']}), "
            f"cross-family(MAE={row['cross_family_MAE']}, RMSE={row['cross_family_RMSE']}, SSIM={row['cross_family_SSIM']}, "
            f"gradient_error={row['cross_family_gradient_error']}, edge_MAE={row['cross_family_edge_MAE']})"
        )
    lines.extend(
        [
            "",
            "## 6. Limitations",
            "- CPU small-sample",
            "- smoke-scale only",
            "- not benchmark-level proof",
            "- geometry is fallback index-based unless real source/receiver metadata are provided",
            "- no DINOv2/SAM/NCS in this run",
            "",
            "## 7. Next Step",
            "- 如果全部 run 成功，进入 V7 boundary auxiliary smoke。",
            "- 如果 geometry/aggregation 出现方向性差异，后续可扩大到 seed=0/1/2。",
            "- 如果无改善，优先检查真实几何 metadata 与 source/receiver/offset encoding。",
        ]
    )
    report = output_root / "protocol_v6_geometry_aggregation_report.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Protocol V6 geometry aggregation smoke report.")
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(write_protocol_v6_geometry_aggregation_report(parse_args().root))


if __name__ == "__main__":
    main()
