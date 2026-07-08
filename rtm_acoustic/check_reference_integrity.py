from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from rtm_acoustic.check_jge_submission_readiness import _section


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "rtm_acoustic" / "docs" / "jge_submission_package"
DEFAULT_MANUSCRIPT = PACKAGE_DIR / "manuscript" / "sci_fwi_rtm_innovation_manuscript_draft.md"
DEFAULT_OUTPUT = PACKAGE_DIR / "JGE_reference_audit.md"

DOI_PATTERN = re.compile(r"https?://doi\.org/(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)")
URL_PATTERN = re.compile(r"https?://\S+")
YEAR_PATTERN = re.compile(r"\((19|20)\d{2}\)")
JOURNAL_PATTERN = re.compile(r"\*[^*\n]+\*")


def parse_references(text: str) -> list[dict[str, Any]]:
    refs = _section(text, "参考文献（待按目标期刊格式重排）")
    entries: list[dict[str, Any]] = []
    for line in refs.splitlines():
        match = re.match(r"^(\d+)\.\s+(.+)$", line.strip())
        if not match:
            continue
        number = int(match.group(1))
        entry = match.group(2)
        doi_match = DOI_PATTERN.search(entry)
        url_match = URL_PATTERN.search(entry)
        entries.append(
            {
                "number": number,
                "entry": entry,
                "has_year": bool(YEAR_PATTERN.search(entry)),
                "has_title_separator": ". " in entry,
                "has_journal_or_venue": bool(JOURNAL_PATTERN.search(entry)),
                "doi": doi_match.group(1) if doi_match else "",
                "url": url_match.group(0).rstrip(".") if url_match else "",
            }
        )
    return entries


def audit_references(manuscript: Path) -> dict[str, Any]:
    text = manuscript.read_text(encoding="utf-8")
    refs = parse_references(text)
    expected = list(range(1, len(refs) + 1))
    actual = [ref["number"] for ref in refs]
    rows: list[dict[str, Any]] = []
    for ref in refs:
        issues: list[str] = []
        if not ref["has_year"]:
            issues.append("missing_year")
        if not ref["has_title_separator"]:
            issues.append("title_or_sentence_separator_unclear")
        if not ref["has_journal_or_venue"]:
            issues.append("missing_italic_venue")
        if not ref["doi"] and not ref["url"]:
            issues.append("missing_doi_or_url")
        rows.append(
            {
                "number": ref["number"],
                "doi": ref["doi"],
                "url": ref["url"],
                "status": "review" if issues else "pass",
                "issues": ";".join(issues),
                "entry": ref["entry"],
            }
        )
    numbering_ok = actual == expected
    return {
        "source": str(manuscript),
        "reference_count": len(refs),
        "numbering_ok": numbering_ok,
        "review_count": sum(1 for row in rows if row["status"] == "review"),
        "ready": numbering_ok and all(row["status"] == "pass" for row in rows),
        "rows": rows,
    }


def write_audit(result: dict[str, Any], output: Path) -> dict[str, Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    json_path = output.with_suffix(".json")
    csv_path = output.with_suffix(".csv")
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    rows = result["rows"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# JGE Reference Audit",
        "",
        f"- `source`: {result['source']}",
        f"- `reference_count`: {result['reference_count']}",
        f"- `numbering_ok`: {result['numbering_ok']}",
        f"- `review_count`: {result['review_count']}",
        f"- `ready`: {result['ready']}",
        "",
        "| No. | DOI | URL | Status | Issues |",
        "|---:|---|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| {row['number']} | {row['doi']} | {row['url']} | {row['status']} | {row['issues']} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `review` means metadata should be checked manually before final submission.",
            "- DOI links and stable publisher, venue, or OpenReview URLs are treated as complete machine-readable locators.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"markdown": output, "json": json_path, "csv": csv_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit manuscript references for numbering and DOI/URL completeness.")
    parser.add_argument("--manuscript", type=Path, default=DEFAULT_MANUSCRIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit_references(args.manuscript)
    written = write_audit(result, args.output)
    for label, path in written.items():
        print(f"{label}: {path}")
    print(f"ready: {result['ready']}")


if __name__ == "__main__":
    main()
