from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _best(rows: list[dict[str, Any]], field: str, *, reverse: bool = False) -> dict[str, Any] | None:
    usable = [row for row in rows if row.get(field) not in {"", None}]
    if not usable:
        return None
    return sorted(usable, key=lambda row: float(row[field]), reverse=reverse)[0]


def _label(row: dict[str, Any] | None) -> str:
    if row is None:
        return "unavailable"
    parts = [row.get("model_name"), row.get("bridge"), row.get("decoder_name"), row.get("loss_name")]
    return " + ".join(str(part) for part in parts if part)


def write_stage_report(
    *,
    output_root: str | Path = Path("outputs/stage_reports"),
    v2_root: str | Path = Path("outputs/protocol_v2_real_cpu_multiseed"),
    v3_root: str | Path = Path("outputs/protocol_v3_selected_multiseed"),
    v4_bridge_root: str | Path = Path("outputs/protocol_v4_bridge_selection"),
    v4_integrated_root: str | Path = Path("outputs/protocol_v4_integrated_bridge_visual_search"),
    v4_fusion_root: str | Path = Path("outputs/protocol_v4_prediction_fusion"),
) -> Path:
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    v3 = _read_csv(Path(v3_root) / "protocol_v3_summary.csv")
    v4_bridge = _read_csv(Path(v4_bridge_root) / "bridge_selection_ranking.csv")
    v4_integrated = _read_csv(Path(v4_integrated_root) / "protocol_v4_integrated_summary.csv")
    v4_fusion = _read_csv(Path(v4_fusion_root) / "protocol_v4_fusion_ranking.csv")
    best_integrated_visual = _best([row for row in v4_integrated if row.get("status") == "SUCCESS"], "visual_score", reverse=True)
    best_integrated_struct = _best([row for row in v4_integrated if row.get("status") == "SUCCESS"], "gradient_error")
    best_fusion = _best([row for row in v4_fusion if row.get("status", "SUCCESS") == "SUCCESS" and str(row.get("reference_only", "")).lower() != "true"], "visual_score", reverse=True)
    best_bridge = v4_bridge[0] if v4_bridge else None
    dino_v3 = [row for row in v3 if "dinov2" in row.get("model_name", "").lower()]
    report_path = out / "protocol_v2_v3_v4_stage_report.md"
    lines = [
        "# Protocol V2/V3/V4 Stage Report",
        "",
        "## Research Goal",
        "主线是 multi-shot gather → seismic-to-vision bridge → Vision FM backbone → cross-shot aggregation → velocity regression head → 2D velocity model。",
        "",
        "## Protocol V2 Summary",
        "Protocol V2 established the cross-family CPU small-sample benchmark path with physical_velocity metrics. Spectrogram bridge showed unstable numerical gains, and DINOv2-LoRA remained a limited probe rather than benchmark evidence.",
        "",
        "## Protocol V3 Summary",
        f"Protocol V3 decoder/loss validation rows={len(v3)}. U-Net decoder consistently improved gradient_error/edge_MAE, simple decoder numerical advantage was not fully stable, and structure_loss reduced structural errors with MAE/RMSE trade-off. DINOv2-LoRA probe rows={len(dino_v3)}.",
        "",
        "## Protocol V4 Bridge Selection",
        f"Bridge selection identified raw_spectrogram as numerically oriented, raw_repeat3 as structurally oriented, and raw_envelope_spectrum3 / spectrogram_multiband as candidates for integrated search. Top bridge by smoke visual_score: {best_bridge.get('bridge_name') if best_bridge else 'unavailable'}.",
        "",
        "## Protocol V4 Integrated Search",
        f"Best visual_score: {_label(best_integrated_visual)}. Best structural metrics: {_label(best_integrated_struct)}. DINOv2-LoRA multi-bridge probe succeeded where available, but remains not benchmark evidence.",
        "",
        "## Protocol V4 Fusion",
        f"Best fusion: {best_fusion.get('fusion_name') if best_fusion else 'unavailable'}. Fusion did not outperform the best single model, and the numerical-structural trade-off remains.",
        "",
        "## Current Best CPU-limited Framework",
        "cnn_baseline + raw_envelope_spectrum3 + unet_decoder + default_l1",
        "",
        "## Key Scientific Finding",
        "The current CPU-limited protocols identify representation, decoder, and loss choices that affect velocity-map quality, but benchmark-level VisionFM claims remain unsupported. Natural-image VisionFM has not formed a stable advantage, and simple fusion did not solve structural recovery.",
        "",
        "## Limitations",
        "- CPU small-sample",
        "- OpenFWI subset",
        "- DINOv2-LoRA limited probe",
        "- not application-level",
        "- not benchmark-level VisionFM claim",
        "",
        "## Next Step",
        "- NCS seismic foundation model probe",
        "- FWI-specific tokenization",
        "- frozen feature cache",
        "- decoder-only training",
        "",
        "The next step is to test seismic-domain pretrained embeddings through NCS-style tokenization and frozen feature probing.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Protocol V2/V3/V4 stage report.")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/stage_reports"))
    parser.add_argument("--v2-root", type=Path, default=Path("outputs/protocol_v2_real_cpu_multiseed"))
    parser.add_argument("--v3-root", type=Path, default=Path("outputs/protocol_v3_selected_multiseed"))
    parser.add_argument("--v4-bridge-root", type=Path, default=Path("outputs/protocol_v4_bridge_selection"))
    parser.add_argument("--v4-integrated-root", type=Path, default=Path("outputs/protocol_v4_integrated_bridge_visual_search"))
    parser.add_argument("--v4-fusion-root", type=Path, default=Path("outputs/protocol_v4_prediction_fusion"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        f"Wrote {write_stage_report(output_root=args.output_root, v2_root=args.v2_root, v3_root=args.v3_root, v4_bridge_root=args.v4_bridge_root, v4_integrated_root=args.v4_integrated_root, v4_fusion_root=args.v4_fusion_root)}"
    )


if __name__ == "__main__":
    main()
