from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_optional(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _exists(repo_root: Path, rel: str) -> bool:
    return (repo_root / rel).exists()


def write_framework_alignment_report(
    *,
    repo_root: str | Path,
    research_plan: str | Path,
    v1_to_v5_report: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    root = Path(repo_root)
    plan_path = Path(research_plan)
    report_path = Path(v1_to_v5_report)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    current_report = _read_optional(report_path)
    research_plan_text = _read_optional(plan_path)
    gap_rows = [
        {"component": "geometry-aware bridge", "status": "implemented" if _exists(root, "data/geometry_aware_bridge.py") else "missing", "priority": "P0", "note": "bridge registry 外挂 geometry embedding"},
        {"component": "source-aware cross-shot aggregation", "status": "implemented" if _exists(root, "models/aggregators.py") else "missing", "priority": "P0", "note": "支持 mean / attention / source-aware attention"},
        {"component": "boundary auxiliary head", "status": "implemented" if _exists(root, "models/boundary_aux_decoder.py") else "missing", "priority": "P0", "note": "boundary_aux_unet + boundary_aux_l1"},
        {"component": "OOD dataset interface", "status": "missing", "priority": "P1", "note": "当前仍缺 Marmousi/Salt/Overthrust 统一接口"},
        {"component": "robustness evaluation", "status": "missing", "priority": "P1", "note": "当前未建立系统鲁棒性评测协议"},
        {"component": "real NCS weight integration", "status": "missing", "priority": "P1", "note": "只有 probe path，没有真实权重接入结果"},
        {"component": "physics consistency loss", "status": "missing", "priority": "P2", "note": "仅有 weak physics / structure-aware 方向，没有 PDE consistency"},
    ]
    gap_table_path = out / "framework_gap_table.csv"
    with gap_table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["component", "status", "priority", "note"])
        writer.writeheader()
        writer.writerows(gap_rows)

    lines = [
        "# Framework Alignment with Research Plan",
        "",
        "## 1. Current Completed Modules",
        "- bridge registry 已部分满足 seismic-to-vision bridge。",
        "- DINOv2-LoRA 已部分满足 natural-image Vision FM transfer。",
        "- Local MAE 已部分满足 seismic-domain pretraining。",
        "- unet_decoder 已部分满足 dense velocity regression decoder。",
        "",
        "## 2. Research Plan Requirements",
        "- geometry-aware seismic-to-vision bridge",
        "- source-aware cross-shot aggregation",
        "- boundary auxiliary velocity decoder",
        "- OOD dataset interface",
        "- robustness evaluation",
        "- real NCS weight integration",
        "- physics consistency loss",
        "",
        "## 3. Matched Components",
        "- bridge registry 已部分满足 seismic-to-vision bridge；",
        "- DINOv2-LoRA 已部分满足 natural-image Vision FM transfer；",
        "- Local MAE 已部分满足 seismic-domain pretraining；",
        "- unet_decoder 已部分满足 dense velocity regression decoder。",
        "",
        "## 4. Missing Components",
        "- geometry-aware bridge；",
        "- source-aware cross-shot aggregation；",
        "- boundary auxiliary head；",
        "- OOD dataset interface；",
        "- robustness evaluation；",
        "- real NCS weight integration；",
        "- physics consistency loss。",
        "",
        "## 5. Optimization Priority",
        "P0:",
        "- geometry embedding",
        "- aggregator registry",
        "- boundary target/loss",
        "",
        "P1:",
        "- DINOv2 frozen feature cache standardization",
        "- PEFT registry",
        "- NCS 2.5D real probe",
        "",
        "P2:",
        "- full SAM/SAMFormer integration",
        "- OOD Marmousi/Salt/Overthrust",
        "- differentiable PDE consistency",
        "",
        "## 6. V6/V7/V8 Roadmap",
        "V6: Geometry-aware bridge + source-aware aggregation。",
        "V7: Boundary auxiliary velocity decoder。",
        "V8: Real NCS / seismic-domain foundation model probe。",
        "",
        "## 7. What Not To Do Now",
        "- 不要盲目扩大自然图像 backbone；",
        "- 不要直接把 SAM 当 velocity regression 主干；",
        "- 不要在 CPU 条件下跑 full OOD matrix；",
        "- 不要把 feasibility evidence 写成 benchmark proof。",
        "",
        "## Notes",
        f"- 已读取 V1–V5 report: {'yes' if bool(current_report) else 'no'}",
        f"- 已读取 research plan: {'yes' if bool(research_plan_text) else 'no'}",
    ]
    report_out = out / "framework_alignment_report.md"
    report_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    roadmap_lines = [
        "# V6/V7/V8 Roadmap",
        "",
        "## V6",
        "- Geometry-aware bridge",
        "- Source-aware cross-shot aggregation",
        "",
        "## V7",
        "- Boundary auxiliary velocity decoder",
        "- Boundary auxiliary loss and diagnostics",
        "",
        "## V8",
        "- Real NCS / seismic-domain foundation model probe",
        "- NCS 2.5D frozen feature path",
    ]
    roadmap_out = out / "v6_v7_v8_roadmap.md"
    roadmap_out.write_text("\n".join(roadmap_lines) + "\n", encoding="utf-8")
    return {"report": report_out, "gap_table": gap_table_path, "roadmap": roadmap_out}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report framework alignment with research plan.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--research-plan", type=Path, required=True)
    parser.add_argument("--v1-to-v5-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    result = write_framework_alignment_report(**vars(parse_args()))
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
