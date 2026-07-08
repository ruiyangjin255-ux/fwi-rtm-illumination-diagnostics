# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def compute_generalization_gaps(in_family: dict[str, Any], cross_family: dict[str, Any]) -> dict[str, float]:
    return {"mae_generalization_gap": round(float(cross_family["mae"])-float(in_family["mae"]),12), "rmse_generalization_gap": round(float(cross_family["rmse"])-float(in_family["rmse"]),12), "ssim_generalization_gap": round(float(in_family["ssim"])-float(cross_family["ssim"]),12), "gradient_generalization_gap": round(float(cross_family["gradient_error"])-float(in_family["gradient_error"]),12), "edge_generalization_gap": round(float(cross_family["edge_mae"])-float(in_family["edge_mae"]),12)}


def analyze_gaps(root: str|Path) -> list[dict[str, Any]]:
    protocol_root=Path(root); rows=[]
    for config_path in sorted(protocol_root.glob("runs/*/*/seed_*/config.json")):
        config=json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("status")!="SUCCESS":continue
        in_metrics=json.loads((config_path.parent/"metrics_in_family_test.json").read_text(encoding="utf-8"));cross=json.loads((config_path.parent/"metrics_cross_family_test.json").read_text(encoding="utf-8"));rows.append({"run_id":config["run_id"],"transfer_id":config["transfer_id"],"method_key":config["method_key"],"seed":int(config["seed"]),**compute_generalization_gaps(in_metrics,cross)})
    fields=list(rows[0]) if rows else ["run_id","transfer_id","method_key","seed"]
    with (protocol_root/"protocol_v13_generalization_gaps.csv").open("w",encoding="utf-8",newline="") as handle:writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    return rows


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--root",required=True);args=parser.parse_args();print(f"gap_rows={len(analyze_gaps(args.root))}")


if __name__=="__main__":main()

