from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RTM_ROOT = ROOT / "rtm_acoustic"
DOCS = RTM_ROOT / "docs"
JGE_REVISION = DOCS / "jge_revision"
FIGURES = DOCS / "jge_main_figures"

JGE_AUTHOR_GUIDE_URL = "https://academic.oup.com/jge/pages/general_instructions"

JGE_PAPER_LIMITS = {
    "abstract_words": 250,
    "keywords": 5,
    "references": 50,
    "figures_tables": 10,
    "word_count": 8000,
    "alt_text_required": True,
}


@dataclass(frozen=True)
class InnovationClaim:
    rank: int
    claim_id: str
    manuscript_claim: str
    novelty_position: str
    required_evidence: str
    implemented_program: str
    primary_outputs: str
    main_figure: str
    claim_boundary: str
    paper_section: str


@dataclass(frozen=True)
class FigureAltText:
    figure: str
    title: str
    alt_text: str
    linked_claim: str


INNOVATION_CLAIMS = [
    InnovationClaim(
        1,
        "quality_gated_fwi_rtm",
        "Quality-gated FWI-to-RTM workflow for salt-model imaging.",
        "Turns weak FWI recovery into a publishable diagnostic workflow by requiring model and RTM image gates before interpretation.",
        "FWI misfit reduction, MAE/RMSE, edge MAE, gradient MAE, update-scale scan, and RTM before/after metrics.",
        "run_optimized_fwi_rtm_pipeline.py; optimize_fwi_update_scale.py; run_rtm_before_after_fwi.py",
        "optimized_fwi_rtm_pipeline_report.json; fwi_update_scale_optimization.csv; rtm_before_after_summary.json",
        "Figure 1; Figure 2",
        "Supports conservative velocity-update screening, not high-resolution FWI velocity recovery.",
        "Methods 3.2; Results 4.1-4.2; Discussion 5.2",
    ),
    InnovationClaim(
        2,
        "target_zone_illumination_diagnostics",
        "Target-zone illumination, RTM-response, and FWI-update energy diagnostics.",
        "Uses salt-top, salt-flank, and subsalt masks to evaluate whether updates and RTM response occur where interpretation needs them.",
        "Source-receiver illumination, low-illumination fraction, RTM response, full update energy, damped update energy.",
        "build_target_zone_illumination_diagnostics.py",
        "target_zone_illumination_metrics.csv; figure5_target_zone_illumination_diagnostics.*",
        "Figure 5",
        "Shows the subsalt target remains weakly illuminated and almost unupdated; it does not claim solved subsalt imaging.",
        "Results 4.4; Discussion 5.1",
    ),
    InnovationClaim(
        3,
        "imaging_condition_separation",
        "Explicit separation of RTM image normalization and FWI gradient preconditioning.",
        "Prevents the common over-claim that better RTM display normalization proves better FWI inversion.",
        "Source-normalized, source-receiver-normalized, Laplacian, receiver illumination, correlation and low-illumination metrics.",
        "run_scheme2_imaging_condition_compare.py; make_jge_main_figures.py",
        "rtm_imaging_condition_metrics.csv; figure3_imaging_condition_diagnostics.*",
        "Figure 3",
        "Supports imaging-condition interpretation; FWI preconditioning remains a separate ablation result.",
        "Methods 3.3-3.4; Results 4.3",
    ),
    InnovationClaim(
        4,
        "negative_ablation_boundary",
        "Negative-result-aware local FWI ablation for illumination preconditioning.",
        "Uses a controlled small salt window to show adaptive line search dominates the current lightweight illumination preconditioner.",
        "Baseline, illumination-preconditioned, epsilon scan, max-update scan, line search, and adaptive extended line-search reductions.",
        "run_small_salt_fwi.py; optimize_existing_salt_result.py",
        "local_fwi_strategy_ranking.csv; adaptive_line_search_summary.json",
        "Figure 4",
        "Positions illumination preconditioning as a limitation and future method target, not as the current main improvement.",
        "Results 4.5; Discussion 5.2",
    ),
    InnovationClaim(
        5,
        "jge_reproducible_submission_pipeline",
        "JGE-oriented reproducible result packaging and readiness checking.",
        "Converts scattered FWI/RTM outputs into journal-count-limited figures, tables, captions, alt text, and package checks.",
        "Automated figure generation, method synthesis, reference audit, readiness report, and package manifest.",
        "make_jge_main_figures.py; build_jge_submission_package.py; check_jge_submission_readiness.py",
        "jge_submission_package_mainfigures; JGE_readiness_report.md; jge_figure_alt_text.md",
        "Figure 1-5; Table 1-5",
        "Checks structure and evidence completeness; journal template formatting and final citation verification remain manual.",
        "Figure plan; Data and code availability",
    ),
]

FIGURE_ALT_TEXT = [
    FigureAltText(
        "Figure 1",
        "Literature-guided FWI-RTM synthesis and claim boundary",
        "Four-panel summary linking literature needs to implemented FWI-RTM quality gates, update-scale selection, evidence strength, and conservative publishable claims.",
        "quality_gated_fwi_rtm",
    ),
    FigureAltText(
        "Figure 2",
        "RTM validation before and after quality-gated FWI updating",
        "RTM images from true, initial, and damped FWI velocities are compared with an after-minus-before panel; the damped update gives only a slight filtered RMSE improvement.",
        "quality_gated_fwi_rtm",
    ),
    FigureAltText(
        "Figure 3",
        "Full-aperture RTM imaging-condition diagnostics",
        "Source-normalized, source-receiver-normalized, Laplacian source-normalized, and receiver-illumination panels show how imaging conditions change salt-model RTM expression.",
        "imaging_condition_separation",
    ),
    FigureAltText(
        "Figure 4",
        "Local FWI ablation and conservative claim boundary",
        "Local salt-window curves and bars show that adaptive line search reduces FWI misfit more than the current lightweight illumination preconditioner.",
        "negative_ablation_boundary",
    ),
    FigureAltText(
        "Figure 5",
        "Target-zone illumination, RTM response, and FWI update diagnostics",
        "Salt-top, salt-flank, and subsalt masks are compared against illumination, RTM response, and FWI update energy, showing weak subsalt response and negligible subsalt updates.",
        "target_zone_illumination_diagnostics",
    ),
]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[field]).replace("\n", " ") for field in fields) + " |")
    return lines


def build(output_dir: Path = JGE_REVISION) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    claims = [asdict(claim) for claim in INNOVATION_CLAIMS]
    alt_text = [asdict(item) for item in FIGURE_ALT_TEXT]
    payload = {
        "jge_author_guide_url": JGE_AUTHOR_GUIDE_URL,
        "jge_paper_limits": JGE_PAPER_LIMITS,
        "innovation_claims": claims,
        "figure_alt_text": alt_text,
    }

    json_path = output_dir / "jge_innovation_framework.json"
    csv_path = output_dir / "jge_innovation_framework.csv"
    md_path = output_dir / "jge_innovation_framework.md"
    alt_json = output_dir / "jge_figure_alt_text.json"
    alt_md = output_dir / "jge_figure_alt_text.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    alt_json.write_text(json.dumps(alt_text, indent=2), encoding="utf-8")
    _write_csv(csv_path, claims)

    md_lines = [
        "# JGE Innovation Framework",
        "",
        f"- Official author guide: {JGE_AUTHOR_GUIDE_URL}",
        "- Paper limits used by the automated framework: abstract <=250 words; keywords <=5; references <=50; figures plus tables <=10; main text <=8000 words.",
        "- Framework purpose: keep each manuscript innovation tied to a runnable program, numeric evidence, a figure/table output, and an explicit claim boundary.",
        "",
        "## Innovation-to-evidence matrix",
        "",
        *_markdown_table(
            claims,
            [
                "rank",
                "claim_id",
                "manuscript_claim",
                "implemented_program",
                "primary_outputs",
                "main_figure",
                "claim_boundary",
            ],
        ),
        "",
        "## Program-level rule",
        "",
        "A claim is manuscript-ready only when it has all four items: runnable script, persisted metric file, figure/table output, and a written claim boundary.",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    alt_lines = [
        "# JGE Figure Alt Text",
        "",
        "OUP/JGE requires alt text for main-article figures. These descriptions are concise and should be placed under the corresponding figure legends during final template conversion.",
        "",
    ]
    for item in FIGURE_ALT_TEXT:
        alt_lines.extend(
            [
                f"## {item.figure}. {item.title}",
                "",
                f"Alt text: {item.alt_text}",
                "",
                f"Linked innovation claim: `{item.linked_claim}`",
                "",
            ]
        )
    alt_md.write_text("\n".join(alt_lines), encoding="utf-8")

    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": md_path,
        "alt_text_json": alt_json,
        "alt_text_markdown": alt_md,
    }


def main() -> None:
    paths = build()
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
