# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.bootstrap_protocol_v12_comparisons import METRICS, paired_bootstrap_metric_deltas
except ModuleNotFoundError:  # direct script execution
    from bootstrap_protocol_v12_comparisons import METRICS, paired_bootstrap_metric_deltas


COMPARISONS={"M3_vs_M2":("dinov2_frozen","random_vit"),"M6_vs_M2":("ncs2d_frozen","random_vit"),"M6_vs_M3":("ncs2d_frozen","dinov2_frozen"),"M5_vs_M6":("spectrogram_dinov2_lora","ncs2d_frozen"),"M4_vs_M3":("dinov2_lora","dinov2_frozen")}


def _load(path:Path)->dict[str,Any]:
    with np.load(path) as payload:return {"prediction":payload["velocity_pred_physical"],"target":payload["velocity_true_physical"],"ids":payload["sample_id"].astype(str).tolist()}


def bootstrap_v13(*,root:str|Path,n_bootstrap:int,comparisons:list[str])->list[dict[str,Any]]:
    protocol_root=Path(root);output=protocol_root/"bootstrap";output.mkdir(exist_ok=True);records={}
    for config_path in protocol_root.glob("runs/*/*/seed_*/config.json"):
        config=json.loads(config_path.read_text(encoding="utf-8"));prediction=config_path.parent/"predictions_cross_family_test.npz"
        if config.get("status")=="SUCCESS" and prediction.is_file():records[(config["transfer_id"],int(config["seed"]),config["method_key"])]=(prediction,config)
    rows=[]
    for comparison_id in comparisons:
        candidate,baseline=COMPARISONS[comparison_id]
        for transfer,seed,method in sorted(records):
            if method!=candidate or (transfer,seed,baseline) not in records:continue
            path,config=records[(transfer,seed,candidate)];base_path,_=records[(transfer,seed,baseline)];result=paired_bootstrap_metric_deltas(**{f"candidate_{k}":v for k,v in _load(path).items()},**{f"baseline_{k}":v for k,v in _load(base_path).items()},n_bootstrap=n_bootstrap,seed=seed+sum(map(ord,comparison_id)));rows.append({"comparison_id":comparison_id,"transfer_id":transfer,"source_family":config["source_family"],"target_family":config["target_family"],"seed":seed,"candidate_method":candidate,"baseline_method":baseline,"n_bootstrap":n_bootstrap,**result})
    fields=["comparison_id","transfer_id","source_family","target_family","seed","candidate_method","baseline_method","n_bootstrap","aligned_sample_count","paired"]+[f"{metric}_{suffix}" for metric in METRICS for suffix in ("mean_difference","ci_low","ci_high","win_probability")]
    with (output/"protocol_v13_bootstrap_deltas.csv").open("w",encoding="utf-8",newline="") as handle:writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    groups=defaultdict(list)
    for row in rows:groups[(row["comparison_id"],row["transfer_id"])].append(row)
    consistency=[]
    for (comparison_id,transfer),group in sorted(groups.items()):
        item={"comparison_id":comparison_id,"transfer_id":transfer,"seed_count":len(group),"aligned_sample_count_min":min(int(row["aligned_sample_count"]) for row in group)}
        for metric in METRICS:item[f"{metric}_improved_seed_count"]=sum(float(row[f"{metric}_mean_difference"])<0 for row in group);item[f"{metric}_ci_below_zero_seed_count"]=sum(float(row[f"{metric}_ci_high"])<0 for row in group)
        consistency.append(item)
    with (output/"protocol_v13_seed_consistency.csv").open("w",encoding="utf-8",newline="") as handle:fields2=list(consistency[0]) if consistency else ["comparison_id","transfer_id"];writer=csv.DictWriter(handle,fieldnames=fields2);writer.writeheader();writer.writerows(consistency)
    lines=["# Protocol V13 预训练来源配对 bootstrap","",f"- bootstrap 次数：{n_bootstrap}","- 所有方法按相同 target sample_id 对齐。","- difference < 0 表示候选误差更低。","","| 比较 | transfer | seed | MAE delta | 95% CI | win probability |","| --- | --- | ---: | ---: | --- | ---: |"]
    for row in rows:lines.append(f"| {row['comparison_id']} | {row['transfer_id']} | {row['seed']} | {row['mae_mean_difference']:.3f} | [{row['mae_ci_low']:.3f}, {row['mae_ci_high']:.3f}] | {row['mae_win_probability']:.3f} |")
    (output/"protocol_v13_bootstrap_summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8");return rows


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--root",required=True);parser.add_argument("--n-bootstrap",type=int,default=2000);parser.add_argument("--comparisons",nargs="+",choices=list(COMPARISONS),default=list(COMPARISONS));args=parser.parse_args();print(f"paired_comparisons={len(bootstrap_v13(root=args.root,n_bootstrap=args.n_bootstrap,comparisons=args.comparisons))}")


if __name__=="__main__":main()
