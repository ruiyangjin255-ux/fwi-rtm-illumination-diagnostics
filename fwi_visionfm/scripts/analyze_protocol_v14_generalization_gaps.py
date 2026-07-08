# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def analyze_protocol_v14_generalization_gaps(*, root: str | Path) -> dict[str, Any]:
    protocol_root = Path(root)
    matrix_path = protocol_root / "protocol_v14_run_matrix.csv"
    rows: list[dict[str, Any]] = []
    if matrix_path.exists():
        with matrix_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    gap_rows = []
    for row in rows:
        gap_rows.append(
            {
                "run_id": row["run_id"],
                "transfer_id": row["transfer_id"],
                "method_id": row["method_id"],
                "bridge_id": row["bridge_id"],
                "seed": row["seed"],
                "generalization_gap_mae": "",
                "generalization_gap_rmse": "",
                "status": row["status"],
            }
        )
    out_path = protocol_root / "protocol_v14_generalization_gaps.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gap_rows[0].keys()) if gap_rows else ["run_id", "transfer_id", "method_id", "bridge_id", "seed", "generalization_gap_mae", "generalization_gap_rmse", "status"])
        writer.writeheader()
        writer.writerows(gap_rows)
    payload = {"row_count": len(gap_rows), "output_path": str(out_path)}
    (protocol_root / "protocol_v14_protocol_integrity_report.md").write_text(
        "# Protocol V14 协议完整性报告\n\n当前已生成 geometry audit、reuse gate 和 run matrix；generalization gap CSV 已占位生成。\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze_protocol_v14_generalization_gaps(root=args.root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
