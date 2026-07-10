from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from admit_fwi.build_jge_innovation_framework import JGE_AUTHOR_GUIDE_URL, JGE_PAPER_LIMITS

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "admit_fwi" / "docs" / "jge_submission_package_mainfigures"
DEFAULT_MANUSCRIPT = PACKAGE_DIR / "manuscript" / "sci_fwi_rtm_innovation_manuscript_draft.md"
DEFAULT_OUTPUT = PACKAGE_DIR / "JGE_readiness_report.md"

JGE_LIMITS = {
    "paper_abstract_words": JGE_PAPER_LIMITS["abstract_words"],
    "paper_keywords": JGE_PAPER_LIMITS["keywords"],
    "paper_references": JGE_PAPER_LIMITS["references"],
    "paper_tables_figures": JGE_PAPER_LIMITS["figures_tables"],
    "paper_word_count": JGE_PAPER_LIMITS["word_count"],
}


def _section(text: str, heading: str) -> str:
    pattern = rf"^## {re.escape(heading)}\s*$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return ""
    start = match.end()
    next_heading = re.search(r"^##\s+", text[start:], flags=re.MULTILINE)
    end = start + next_heading.start() if next_heading else len(text)
    return text[start:end].strip()


def _word_count(text: str) -> int:
    ascii_words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:\.\d+)?%?", text)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    return len(ascii_words) + len(cjk_chars)


def _english_word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?|\d+(?:\.\d+)?%?", text))


def _keyword_count(text: str) -> int:
    match = re.search(r"\*\*Keywords:\*\*\s*(.+)", text)
    if not match:
        match = re.search(r"\*\*关键词：\*\*\s*(.+)", text)
    if not match:
        return 0
    raw = match.group(1)
    return len([item.strip() for item in re.split(r";|；|,", raw) if item.strip()])


def _reference_count(text: str) -> int:
    refs = _section(text, "参考文献（待按目标期刊格式重排）")
    return len(re.findall(r"^\d+\.\s+", refs, flags=re.MULTILINE))


def _figure_table_count(text: str) -> int:
    section = _section(text, "8 图表计划")
    if not section:
        return 0
    return len(re.findall(r"^\|\s*(?:图|表)\s*\d+", section, flags=re.MULTILINE))


def _status(ok: bool) -> str:
    return "pass" if ok else "review"


def check_manuscript(manuscript: Path, package_dir: Path = PACKAGE_DIR) -> dict[str, Any]:
    text = manuscript.read_text(encoding="utf-8")
    abstract = _section(text, "Abstract")
    chinese_abstract = _section(text, "摘要")
    body_words = _word_count(text)
    english_abstract_words = _english_word_count(abstract)
    keywords = _keyword_count(text)
    references = _reference_count(text)
    figure_tables = _figure_table_count(text)
    required_files = [
        package_dir / "package_manifest.json",
        package_dir / "package_file_index.csv",
        package_dir / "JGE_submission_checklist.md",
        package_dir / "JGE_reference_audit.md",
        package_dir / "reports" / "jge_innovation_framework.md",
        package_dir / "figures" / "jge_figure_alt_text.md",
        package_dir / "figures" / "figure1_fwi_quality_gate.tiff",
        package_dir / "figures" / "figure2_rtm_before_after_validation.tiff",
        package_dir / "figures" / "figure3_imaging_condition_diagnostics.tiff",
        package_dir / "figures" / "figure4_spatial_update_gate.tiff",
        package_dir / "figures" / "figure5_target_zone_illumination_diagnostics.tiff",
        package_dir / "tables" / "fwi_update_scale_optimization.csv",
        package_dir / "tables" / "spatial_update_gate_candidates.csv",
        package_dir / "tables" / "target_zone_illumination_metrics.csv",
        package_dir / "reports" / "optimized_fwi_rtm_pipeline_report.md",
    ]

    checks = [
        {
            "item": "English abstract <=250 words",
            "value": english_abstract_words,
            "limit": JGE_LIMITS["paper_abstract_words"],
            "status": _status(english_abstract_words <= JGE_LIMITS["paper_abstract_words"]),
        },
        {
            "item": "Keywords <=5",
            "value": keywords,
            "limit": JGE_LIMITS["paper_keywords"],
            "status": _status(0 < keywords <= JGE_LIMITS["paper_keywords"]),
        },
        {
            "item": "References <=50",
            "value": references,
            "limit": JGE_LIMITS["paper_references"],
            "status": _status(references <= JGE_LIMITS["paper_references"]),
        },
        {
            "item": "Figures+tables <=10",
            "value": figure_tables,
            "limit": JGE_LIMITS["paper_tables_figures"],
            "status": _status(figure_tables <= JGE_LIMITS["paper_tables_figures"]),
        },
        {
            "item": "Approx manuscript word count <=8000",
            "value": body_words,
            "limit": JGE_LIMITS["paper_word_count"],
            "status": _status(body_words <= JGE_LIMITS["paper_word_count"]),
        },
        {
            "item": "Chinese abstract present",
            "value": bool(chinese_abstract),
            "limit": True,
            "status": _status(bool(chinese_abstract)),
        },
        {
            "item": "Data/code availability statement present",
            "value": "数据与代码可用性声明" in text,
            "limit": True,
            "status": _status("数据与代码可用性声明" in text),
        },
        {
            "item": "AI assistance statement present",
            "value": "AI 辅助声明" in text,
            "limit": True,
            "status": _status("AI 辅助声明" in text),
        },
        {
            "item": "Innovation framework referenced in manuscript",
            "value": "创新点 1：照明可信域空间 FWI 更新门控" in text,
            "limit": True,
            "status": _status("创新点 1：照明可信域空间 FWI 更新门控" in text),
        },
    ]
    for file_path in required_files:
        checks.append(
            {
                "item": f"Package file exists: {file_path.relative_to(package_dir)}",
                "value": file_path.exists(),
                "limit": True,
                "status": _status(file_path.exists()),
            }
        )
    return {
        "source": str(manuscript),
        "guideline_source": f"OUP Journal of Geophysics and Engineering Instructions to Authors: {JGE_AUTHOR_GUIDE_URL}",
        "limits": JGE_LIMITS,
        "checks": checks,
        "ready": all(check["status"] == "pass" for check in checks),
    }


def write_report(result: dict[str, Any], output: Path) -> dict[str, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = [
        "# JGE Readiness Report",
        "",
        f"- `source`: {result['source']}",
        f"- `guideline_source`: {result['guideline_source']}",
        f"- `ready`: {result['ready']}",
        "",
        "| Check | Value | Limit | Status |",
        "|---|---:|---:|---|",
    ]
    for check in result["checks"]:
        lines.append(f"| {check['item']} | {check['value']} | {check['limit']} | {check['status']} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This is an automated structural check, not a substitute for journal template formatting or reference verification.",
            "- JGE/OUP author instructions can change; recheck the official page immediately before submission.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"markdown": output, "json": json_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check the current package against JGE Paper limits.")
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--package-dir", type=Path, default=PACKAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = check_manuscript(args.manuscript, package_dir=args.package_dir)
    written = write_report(result, args.output)
    for label, path in written.items():
        print(f"{label}: {path}")
    print(f"ready: {result['ready']}")


if __name__ == "__main__":
    main()
