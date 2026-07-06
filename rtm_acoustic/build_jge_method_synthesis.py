from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RTM_ROOT = ROOT / "rtm_acoustic"
OUT_DIR = RTM_ROOT / "docs" / "jge_revision"


METHOD_SYNTHESIS_ROWS = [
    {
        "literature_direction": "Low-frequency and low-wavenumber FWI recovery",
        "representative_sources": "Virieux & Operto 2009; JGE SDCI FWI 2024",
        "problem_for_current_result": "The full salt FWI reduces data misfit but does not recover a sharper salt boundary.",
        "implemented_response": "Do not claim high-quality FWI; add model-quality and edge-error gates before RTM.",
        "paper_role": "claim boundary and quality-control mechanism",
    },
    {
        "literature_direction": "Regularized or constrained FWI",
        "representative_sources": "JGE sparse regularization FWI 2024; high-order TV FWI 2021",
        "problem_for_current_result": "Unregularized updates mainly improve non-edge MAE and can degrade gradient/edge metrics.",
        "implemented_response": "Rank alpha candidates by MAE, RMSE, edge MAE, gradient MAE and reject harmful updates.",
        "paper_role": "diagnostic update-scale selection",
    },
    {
        "literature_direction": "Optimization and step-length control",
        "representative_sources": "JGE augmented-Lagrangian FWI 2024; Devito FWI optimization examples",
        "problem_for_current_result": "Simple illumination scaling is weaker than adaptive step selection in the local salt window.",
        "implemented_response": "Use local-window line-search ablation as mechanistic evidence.",
        "paper_role": "strongest FWI-side experimental evidence",
    },
    {
        "literature_direction": "RTM/LSRTM imaging-condition improvement",
        "representative_sources": "JGE gradient LSRTM 2016; local cross-correlation LSRTM 2022; Wasserstein LSRTM 2026",
        "problem_for_current_result": "The current code is RTM diagnostics, not LSRTM or high-resolution reflectivity inversion.",
        "implemented_response": "Report source, source-receiver, Laplacian and illumination metrics without claiming LSRTM.",
        "paper_role": "RTM-side quantitative imaging-condition comparison",
    },
    {
        "literature_direction": "Reproducible benchmark and data-driven FWI",
        "representative_sources": "OpenFWI 2022; DINOv2 2024 as a possible feature prior",
        "problem_for_current_result": "The ML/foundation-model branch is smoke-level and cannot support the main result.",
        "implemented_response": "Keep ML priors as a future extension; keep this paper physics-first and reproducible.",
        "paper_role": "future work only",
    },
]


def build(output_dir: Path = OUT_DIR) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "method_synthesis_matrix.csv"
    md_path = output_dir / "method_synthesis_matrix.md"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(METHOD_SYNTHESIS_ROWS[0]))
        writer.writeheader()
        writer.writerows(METHOD_SYNTHESIS_ROWS)

    lines = [
        "# Literature-guided method synthesis matrix",
        "",
        "This matrix reframes the paper away from a weak standalone FWI claim and toward a defensible integrated FWI-RTM diagnostic workflow.",
        "",
        "| Literature direction | Current problem | Implemented response | Paper role |",
        "|---|---|---|---|",
    ]
    for row in METHOD_SYNTHESIS_ROWS:
        lines.append(
            "| {literature_direction} | {problem_for_current_result} | {implemented_response} | {paper_role} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Revised innovation statement",
            "",
            "The manuscript should not claim a new high-performance FWI algorithm. Its defensible contribution is a literature-guided, reproducible diagnostic workflow that combines FWI misfit reduction, model-structure quality gates, update-scale rejection, RTM before/after validation, imaging-condition diagnostics and local optimizer ablation.",
            "",
            "## Source links used for positioning",
            "",
            "- JGE SDCI FWI: https://academic.oup.com/jge/article/21/6/1594/7762962",
            "- JGE LSRTM local cross-correlation: https://academic.oup.com/jge/article/19/3/376/6597025",
            "- JGE LSRTM gradient improvement: https://academic.oup.com/jge/article/13/2/172/5113404",
            "- JGE adaptive Wasserstein LSRTM: https://academic.oup.com/jge/advance-article/doi/10.1093/jge/gxag083/8708462",
            "- Virieux and Operto FWI overview: https://doi.org/10.1190/1.3238367",
            "- OpenFWI benchmark: https://proceedings.neurips.cc/paper_files/paper/2022/hash/27d3ef263c7cb8d542c4f9815a49b69b-Abstract-Datasets_and_Benchmarks.html",
            "- DINOv2: https://arxiv.org/abs/2304.07193",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"csv": csv_path, "markdown": md_path}


def main() -> None:
    paths = build()
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
