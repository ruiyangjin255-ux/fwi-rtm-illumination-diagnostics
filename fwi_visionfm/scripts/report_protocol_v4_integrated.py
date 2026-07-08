from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

from fwi_visionfm.scripts.summarize_protocol_v4_integrated import write_integrated_summary


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _success(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("status") == "SUCCESS"]


def _best(rows: list[dict[str, Any]], field: str, *, reverse: bool = False) -> dict[str, Any] | None:
    usable = [row for row in rows if row.get(field) != ""]
    if not usable:
        return None
    return sorted(usable, key=lambda row: float(row[field]), reverse=reverse)[0]


def _label(row: dict[str, Any] | None) -> str:
    if row is None:
        return "unavailable"
    return f"{row['model_name']} + {row['bridge']} + {row['decoder_name']} + {row['loss_name']} + seed {row['seed']}"


def _copy_grid(row: dict[str, Any] | None, root: Path, name: str) -> None:
    if row is None:
        return
    if not row.get("run_dir"):
        return
    src = Path(row["run_dir"]) / "best_prediction_grid.png"
    if src.exists():
        shutil.copyfile(src, root / name)


def _bridge_best(rows: list[dict[str, Any]], bridge: str, field: str, *, reverse: bool = False) -> dict[str, Any] | None:
    return _best([row for row in rows if row.get("bridge") == bridge], field, reverse=reverse)


def write_integrated_report(root: str | Path) -> Path:
    output_root = Path(root)
    summary_path = output_root / "protocol_v4_integrated_summary.csv"
    if not summary_path.exists():
        summary_path = write_integrated_summary(output_root)
    rows = _read_rows(summary_path)
    success = _success(rows)
    best_visual = _best(success, "visual_score", reverse=True)
    best_mae = _best(success, "MAE")
    best_rmse = _best(success, "RMSE")
    best_grad = _best(success, "gradient_error")
    best_edge = _best(success, "edge_MAE")
    best_raw_spec = _bridge_best(success, "raw_spectrogram", "visual_score", reverse=True)
    best_multi = _bridge_best(success, "spectrogram_multiband", "visual_score", reverse=True)
    best_raw = _bridge_best(success, "raw_repeat3", "gradient_error")
    best_hybrid = _bridge_best(success, "raw_envelope_spectrum3", "visual_score", reverse=True)
    dino_rows = [row for row in success if row.get("model_name") == "dinov2_lora_smoke"]
    _copy_grid(best_visual, output_root, "best_by_visual_score_prediction_grid.png")
    _copy_grid(best_grad, output_root, "best_by_gradient_error_prediction_grid.png")
    _copy_grid(best_mae, output_root, "best_by_MAE_prediction_grid.png")
    report_path = output_root / "protocol_v4_integrated_report.md"
    lines = [
        "# Integrated Protocol V4 Bridge Visual Search Report",
        "",
        "Integrated Protocol V4 evaluates whether bridge-level advantages persist when coupled with backbone, decoder, and loss choices. It provides a CPU-efficient pathway to select the best available representation-model combination for final velocity-map quality, while keeping application-level claims conservative.",
        "",
        "## Best Configurations",
        f"- Best visual_score: {_label(best_visual)}; visual_score={best_visual.get('visual_score') if best_visual else ''}",
        f"- Best MAE: {_label(best_mae)}; MAE={best_mae.get('MAE') if best_mae else ''}",
        f"- Best RMSE: {_label(best_rmse)}; RMSE={best_rmse.get('RMSE') if best_rmse else ''}",
        f"- Best gradient_error: {_label(best_grad)}; gradient_error={best_grad.get('gradient_error') if best_grad else ''}",
        f"- Best edge_MAE: {_label(best_edge)}; edge_MAE={best_edge.get('edge_MAE') if best_edge else ''}",
        "",
        "## Bridge Questions",
        f"- Highest visual bridge in full chain: {best_visual.get('bridge') if best_visual else 'unavailable'}.",
        f"- Bridge-selection smoke top bridge still best: {best_visual.get('bridge') in {'raw_spectrogram', 'spectrogram_multiband'} if best_visual else 'unavailable'}.",
        f"- raw_spectrogram numerical tendency: best available visual row is {_label(best_raw_spec)}.",
        f"- raw_repeat3 structural control retained: {_label(best_raw)}.",
        f"- spectrogram_multiband vs raw_spectrogram visual_score: {best_multi.get('visual_score') if best_multi else ''} vs {best_raw_spec.get('visual_score') if best_raw_spec else ''}.",
        f"- raw_envelope_spectrum3 compromise candidate: {_label(best_hybrid)}; visual_score={best_hybrid.get('visual_score') if best_hybrid else ''}.",
        f"- DINOv2-LoRA bridge probe rows: {len(dino_rows)}; this remains a probe, not benchmark evidence.",
        "",
        "## Limitation",
        "Current outputs are not application-level and do not establish complex FWI deployment quality.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report integrated Protocol V4 results.")
    parser.add_argument("--root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(f"Wrote {write_integrated_report(parse_args().root)}")


if __name__ == "__main__":
    main()
