from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from rtm_acoustic.build_jge_innovation_framework import build as build_innovation_framework
from rtm_acoustic.build_jge_method_synthesis import build as build_method_synthesis
from rtm_acoustic.build_spatial_update_gate import build as build_spatial_update_gate
from rtm_acoustic.build_target_zone_illumination_diagnostics import build as build_target_zone_diagnostics
from rtm_acoustic.check_jge_submission_readiness import check_manuscript, write_report
from rtm_acoustic.check_reference_integrity import audit_references, write_audit


ROOT = Path(__file__).resolve().parents[1]
RTM_ROOT = ROOT / "rtm_acoustic"
DOCS = RTM_ROOT / "docs"
FIGURES = DOCS / "jge_main_figures"
JGE_REVISION = DOCS / "jge_revision"
FWI_RUN = RTM_ROOT / "outputs" / "FWI" / "full_salt_fwi_cg_allshots_v2"
PIPELINE_DIR = FWI_RUN / "optimized_fwi_rtm_pipeline"
SCHEME2_DIR = RTM_ROOT / "outputs" / "RTM" / "seg_salt_scheme2_full30m_nt4001_workers4"
DEFAULT_OUTPUT = DOCS / "jge_submission_package_mainfigures"

MANUSCRIPT = DOCS / "sci_fwi_rtm_innovation_manuscript_draft.md"
CAPTIONS = FIGURES / "jge_main_figure_captions.md"
PIPELINE_REPORT = PIPELINE_DIR / "optimized_fwi_rtm_pipeline_report.md"

FIGURE_STEMS = {
    "figure1_fwi_quality_gate": "figure1_fwi_quality_gate",
    "figure2_rtm_before_after_validation": "figure2_rtm_before_after_validation",
    "figure3_imaging_condition_diagnostics": "figure3_imaging_condition_diagnostics",
    "figure4_spatial_update_gate": "figure4_spatial_update_gate",
    "figure5_target_zone_illumination_diagnostics": "figure5_target_zone_illumination_diagnostics",
}

TABLE_FILES = [
    "full_fwi_ranking.csv",
    "fwi_model_quality_ranking.csv",
    "fwi_update_scale_optimization.csv",
    "rtm_imaging_condition_metrics.csv",
    "local_fwi_strategy_ranking.csv",
    "spatial_update_gate_candidates.csv",
    "innovation_ranking.csv",
    "method_synthesis_matrix.csv",
    "target_zone_illumination_metrics.csv",
]

TARGET_ZONE_INPUTS = [
    FWI_RUN / "full_salt_true_model.npy",
    FWI_RUN / "full_salt_initial_model.npy",
    FWI_RUN / "full_salt_inverted_model.npy",
    FIGURES / "figure5_target_zone_illumination_diagnostics.png",
    JGE_REVISION / "target_zone_illumination_metrics.csv",
]

SPATIAL_GATE_PREBUILT = [
    FIGURES / "figure4_spatial_update_gate.png",
    JGE_REVISION / "spatial_update_gate_candidates.csv",
]


def _copy_file(src: Path, dst: Path) -> dict[str, Any]:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "source": str(src),
        "package_path": str(dst),
        "bytes": dst.stat().st_size,
    }


def _copy_pipeline_report(src: Path, dst: Path) -> dict[str, Any]:
    if PIPELINE_REPORT.exists():
        return _copy_file(src, dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Optimized FWI-RTM Pipeline Report",
        "",
        "This lightweight public repository does not include raw FWI/RTM output arrays or full run logs.",
        "Use the packaged CSV tables, main figures, manuscript draft, and innovation framework for manuscript review.",
        "Re-run the full local pipeline with `python -m rtm_acoustic.run_optimized_fwi_rtm_pipeline --run-rtm` when the raw SEG/Salt-derived data products are available.",
    ]
    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "source": str(PIPELINE_REPORT),
        "package_path": str(dst),
        "bytes": dst.stat().st_size,
        "note": "placeholder written because full local pipeline report was not present",
    }


def _read_pipeline_metrics() -> dict[str, Any]:
    path = PIPELINE_DIR / "optimized_fwi_rtm_pipeline_report.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        rtm = data.get("rtm_validation", {})
        return {
            "selected_alpha": data.get("update_scale", {}).get("selected_alpha"),
            "rtm_verdict": rtm.get("verdict"),
            "filtered_rmse_before": rtm.get("filtered_rmse_before"),
            "filtered_rmse_after": rtm.get("filtered_rmse_after"),
            "filtered_rmse_improvement_fraction": rtm.get("filtered_rmse_improvement_fraction"),
        }
    scale_csv = JGE_REVISION / "fwi_update_scale_optimization.csv"
    if scale_csv.exists():
        with scale_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("selected") == "True":
                    return {
                        "selected_alpha": float(row["alpha"]),
                        "rtm_verdict": None,
                        "filtered_rmse_before": None,
                        "filtered_rmse_after": None,
                        "filtered_rmse_improvement_fraction": None,
                        "source": str(scale_csv),
                    }
    return {}


def _can_rebuild_target_zone_diagnostics() -> bool:
    data_inputs = [
        FWI_RUN / "full_salt_true_model.npy",
        FWI_RUN / "full_salt_initial_model.npy",
        FWI_RUN / "full_salt_inverted_model.npy",
        SCHEME2_DIR / "scheme2_source_illumination.npy",
        SCHEME2_DIR / "scheme2_receiver_illumination.npy",
    ]
    return all(path.exists() for path in data_inputs)


def _has_prebuilt_target_zone_diagnostics() -> bool:
    prebuilt = [
        FIGURES / "figure5_target_zone_illumination_diagnostics.png",
        JGE_REVISION / "target_zone_illumination_metrics.csv",
    ]
    return all(path.exists() for path in prebuilt)


def _has_prebuilt_spatial_update_gate() -> bool:
    return all(path.exists() for path in SPATIAL_GATE_PREBUILT)


def _refresh_data_dependent_results() -> None:
    if _can_rebuild_target_zone_diagnostics():
        build_spatial_update_gate()
    elif not _has_prebuilt_spatial_update_gate():
        missing = [str(path) for path in SPATIAL_GATE_PREBUILT if not path.exists()]
        raise FileNotFoundError(
            "Spatial update gate cannot be rebuilt and no prebuilt figure/table pair was found. "
            f"Missing outputs include: {missing}"
        )
    if _can_rebuild_target_zone_diagnostics():
        build_target_zone_diagnostics()
    elif not _has_prebuilt_target_zone_diagnostics():
        missing = [str(path) for path in TARGET_ZONE_INPUTS if not path.exists()]
        raise FileNotFoundError(
            "Target-zone diagnostics cannot be rebuilt and no prebuilt figure/table pair was found. "
            f"Missing inputs include: {missing}"
        )


def write_submission_checklist(path: Path, manifest: dict[str, Any]) -> None:
    metrics = manifest.get("pipeline_metrics", {})
    lines = [
        "# JGE Submission Checklist",
        "",
        "## Ready items",
        "",
        "- Manuscript draft is included in Markdown format.",
        "- Figure captions are included separately.",
        "- Figure alt text is included for final OUP/JGE template conversion.",
        "- Core result tables are included as CSV files.",
        "- Innovation claims are mapped to programs, evidence files, figures, and claim boundaries.",
        "- Figure 1-5 files are packaged in the requested formats.",
        "- Optimized FWI-RTM pipeline report is included.",
        "",
        "## Key result gate",
        "",
        f"- `selected_alpha`: {metrics.get('selected_alpha')}",
        f"- `rtm_verdict`: {metrics.get('rtm_verdict')}",
        f"- `filtered_rmse_before`: {metrics.get('filtered_rmse_before')}",
        f"- `filtered_rmse_after`: {metrics.get('filtered_rmse_after')}",
        f"- `filtered_rmse_improvement_fraction`: {metrics.get('filtered_rmse_improvement_fraction')}",
        "",
        "## Manual checks before journal submission",
        "",
        "- Convert the manuscript to the target JGE/OUP Word or LaTeX template.",
        "- Recheck JGE word limit, abstract length, keywords, figure count, and reference style against the current author guidelines.",
        "- Verify every DOI, page number, author spelling, and in-text citation.",
        "- Confirm that SEG/Salt model redistribution is allowed before uploading any raw model data.",
        "- Keep claims conservative: the result supports a diagnostic quality-gated FWI-RTM workflow, not production-grade FWI velocity recovery.",
        "- If using TIFF figures, confirm final journal-required DPI after layout scaling.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_package(
    output_dir: Path,
    *,
    figure_formats: list[str],
) -> dict[str, Any]:
    build_innovation_framework()
    build_method_synthesis()
    _refresh_data_dependent_results()

    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "figure_formats": figure_formats,
        "files": {},
        "pipeline_metrics": _read_pipeline_metrics(),
    }

    manifest["files"]["manuscript"] = _copy_file(MANUSCRIPT, output_dir / "manuscript" / MANUSCRIPT.name)
    manifest["files"]["captions"] = _copy_file(CAPTIONS, output_dir / "figures" / CAPTIONS.name)
    manifest["files"]["pipeline_report"] = _copy_pipeline_report(
        PIPELINE_REPORT,
        output_dir / "reports" / PIPELINE_REPORT.name,
    )
    manifest["files"]["jge_upgrade_plan"] = _copy_file(
        JGE_REVISION / "jge_upgrade_plan.md",
        output_dir / "reports" / "jge_upgrade_plan.md",
    )
    manifest["files"]["literature_positioning"] = _copy_file(
        JGE_REVISION / "jge_literature_positioning.md",
        output_dir / "reports" / "jge_literature_positioning.md",
    )
    manifest["files"]["method_synthesis"] = _copy_file(
        JGE_REVISION / "method_synthesis_matrix.md",
        output_dir / "reports" / "method_synthesis_matrix.md",
    )
    manifest["files"]["innovation_framework"] = _copy_file(
        JGE_REVISION / "jge_innovation_framework.md",
        output_dir / "reports" / "jge_innovation_framework.md",
    )
    manifest["files"]["figure_alt_text"] = _copy_file(
        JGE_REVISION / "jge_figure_alt_text.md",
        output_dir / "figures" / "jge_figure_alt_text.md",
    )
    manifest["files"]["core_innovation_from_references"] = _copy_file(
        JGE_REVISION / "core_innovation_from_references.md",
        output_dir / "reports" / "core_innovation_from_references.md",
    )

    figure_entries: dict[str, Any] = {}
    for package_stem, source_stem in FIGURE_STEMS.items():
        for ext in figure_formats:
            src = FIGURES / f"{source_stem}.{ext}"
            dst = output_dir / "figures" / f"{package_stem}.{ext}"
            figure_entries[f"{package_stem}.{ext}"] = _copy_file(src, dst)
    manifest["files"]["figures"] = figure_entries

    table_entries: dict[str, Any] = {}
    for name in TABLE_FILES:
        table_entries[name] = _copy_file(JGE_REVISION / name, output_dir / "tables" / name)
    table_entries["jge_innovation_framework.csv"] = _copy_file(
        JGE_REVISION / "jge_innovation_framework.csv",
        output_dir / "tables" / "jge_innovation_framework.csv",
    )
    manifest["files"]["tables"] = table_entries

    checklist_path = output_dir / "JGE_submission_checklist.md"
    write_submission_checklist(checklist_path, manifest)
    manifest["files"]["checklist"] = {
        "package_path": str(checklist_path),
        "bytes": checklist_path.stat().st_size,
    }

    manifest_path = output_dir / "package_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)

    summary_csv = output_dir / "package_file_index.csv"
    rows: list[dict[str, Any]] = []
    for group, value in manifest["files"].items():
        if isinstance(value, dict) and "package_path" in value:
            rows.append({"group": group, "name": Path(value["package_path"]).name, **value})
        elif isinstance(value, dict):
            for name, entry in value.items():
                rows.append({"group": group, "name": name, **entry})
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        preferred_fields = ["group", "name", "source", "package_path", "bytes", "note"]
        extra_fields = sorted({field for row in rows for field in row} - set(preferred_fields))
        writer = csv.DictWriter(handle, fieldnames=preferred_fields + extra_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    manifest["file_index_path"] = str(summary_csv)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    reference_result = audit_references(output_dir / "manuscript" / MANUSCRIPT.name)
    reference_paths = write_audit(reference_result, output_dir / "JGE_reference_audit.md")
    manifest["files"]["reference_audit"] = {
        key: str(value) for key, value in reference_paths.items()
    }
    readiness_result = check_manuscript(output_dir / "manuscript" / MANUSCRIPT.name, package_dir=output_dir)
    readiness_paths = write_report(readiness_result, output_dir / "JGE_readiness_report.md")
    manifest["files"]["readiness_report"] = {
        key: str(value) for key, value in readiness_paths.items()
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a JGE-oriented submission package from current FWI-RTM outputs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--figure-formats",
        nargs="+",
        default=["png", "pdf", "svg", "tiff"],
        choices=["png", "pdf", "svg", "tiff"],
        help="Figure formats to copy into the package.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_package(args.output_dir, figure_formats=list(args.figure_formats))
    print(f"manifest: {manifest['manifest_path']}")
    print(f"file_index: {manifest['file_index_path']}")
    print(f"checklist: {manifest['files']['checklist']['package_path']}")


if __name__ == "__main__":
    main()
