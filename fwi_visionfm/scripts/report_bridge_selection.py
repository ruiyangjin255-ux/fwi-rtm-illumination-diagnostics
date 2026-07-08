from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from fwi_visionfm.scripts.summarize_bridge_selection import write_bridge_selection_summary


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _best(rows: list[dict[str, Any]], field: str, *, higher: bool = False) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(rows, key=lambda row: float(row[field]) if row.get(field) != "" else (-1.0 if higher else float("inf")), reverse=higher)[0]


def _line(label: str, row: dict[str, Any] | None, metric: str) -> str:
    if row is None:
        return f"- {label}: unavailable"
    return f"- {label}: {row['bridge_name']} ({metric}={row.get(metric, '')})"


def write_bridge_selection_report(root: str | Path) -> Path:
    output_root = Path(root)
    paths = write_bridge_selection_summary(output_root)
    rows = _read_rows(paths["summary"])
    success = [row for row in rows if row.get("status") == "SUCCESS"]
    ranking = _read_rows(paths["ranking"])
    best_mae = _best(success, "MAE")
    best_rmse = _best(success, "RMSE")
    best_grad = _best(success, "gradient_error")
    best_edge = _best(success, "edge_MAE")
    best_visual = _best(success, "visual_score", higher=True)
    top = ranking[:3]
    numerical_gain_without_structural = False
    if best_mae and best_grad and best_mae["bridge_name"] != best_grad["bridge_name"]:
        numerical_gain_without_structural = True
    report_path = output_root / "bridge_selection_report.md"
    lines = [
        "# Protocol V4 Bridge Selection Report",
        "",
        "Bridge auto-selection provides a CPU-efficient way to compare raw, envelope, spectrogram, and hybrid seismic-to-vision representations. It helps identify which input representation better supports numerical accuracy or structural recovery before scaling to larger VisionFM experiments.",
        "",
        "## Best Metrics",
        _line("Best MAE", best_mae, "MAE"),
        _line("Best RMSE", best_rmse, "RMSE"),
        _line("Best gradient_error", best_grad, "gradient_error"),
        _line("Best edge_MAE", best_edge, "edge_MAE"),
        _line("Best visual_score", best_visual, "visual_score"),
        "",
        "## Representation Pattern",
        "raw、envelope、spectrogram 与 hybrid bridge 在当前 smoke 中表现出不同数值/结构取向，需以 visual_score 和结构指标共同筛选。",
        "",
        "## Numerical Gain Without Structural Gain",
        "Detected." if numerical_gain_without_structural else "Not detected in available rows.",
        "",
        "## Recommended Top Bridges",
        *[f"- top {index}: {row['bridge_name']} (visual_score={row['visual_score']})" for index, row in enumerate(top, start=1)],
        "",
        "## Limitation",
        "当前 bridge selection 是 CPU small-sample smoke，距离复杂 FWI 应用级成图仍有明显验证缺口。",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report Protocol V4 bridge selection.")
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(f"Wrote {write_bridge_selection_report(parse_args().root)}")


if __name__ == "__main__":
    main()
