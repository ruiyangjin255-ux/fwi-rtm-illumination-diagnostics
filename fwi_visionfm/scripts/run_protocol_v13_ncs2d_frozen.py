# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fwi_visionfm.datasets import load_npz_sample
from fwi_visionfm.evaluation.metrics import compute_velocity_metrics
from fwi_visionfm.models.protocol_v11_model_registry import build_protocol_v11_model
from fwi_visionfm.models.seismic_backbones.ncs_backbone import load_ncs_model
from fwi_visionfm.models.seismic_bridge import SeismicToVisionBridge
from fwi_visionfm.torch_backend import require_torch_backend
try:
    from scripts.run_protocol_v11_visionfm_crossfamily import _read_bounds, _write_grid, _write_history
    from scripts.run_protocol_v12_spectrogram_dinov2_confirmation import build_optimizer_with_registration_report
except ModuleNotFoundError:  # direct script execution
    from run_protocol_v11_visionfm_crossfamily import _read_bounds, _write_grid, _write_history
    from run_protocol_v12_spectrogram_dinov2_confirmation import build_optimizer_with_registration_report


REQUIRED = {"config.json", "config_hash.txt", "model_card.json", "feature_cache_metadata.json", "train_history.csv", "metrics_val.json", "metrics_in_family_test.json", "metrics_cross_family_test.json", "predictions_in_family_test.npz", "predictions_cross_family_test.npz", "prediction_grid.png", "gradient_grid.png", "run_log.txt"}


def _bool(value: Any) -> bool:
    return str(value).lower() not in {"0", "false", "no", "off", ""}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def validate_real_feature_cache(path: str | Path) -> dict[str, Any]:
    cache = Path(path)
    with np.load(cache, allow_pickle=False) as payload:
        required = {"features", "target", "sample_id", "is_real_feature"}
        if not required.issubset(payload.files): raise ValueError("real NCS2D cache contract is incomplete")
        is_real = bool(np.asarray(payload["is_real_feature"]).item())
        if not is_real: raise ValueError("real NCS2D feature is required; fallback cache rejected")
        return {"path": str(cache), "is_real_feature": True, "sample_count": int(len(payload["features"])), "feature_shape": list(payload["features"].shape), "target_shape": list(payload["target"].shape), "sample_ids_hash": _hash(payload["sample_id"].astype(str).tolist()), "sha256": hashlib.sha256(cache.read_bytes()).hexdigest()}


def _ids(paths: list[Path]) -> list[str]:
    return [f"{path.parent.name}/{path.name}" for path in paths]


def _paths(rows: list[dict[str, Any]]) -> list[Path]:
    return [Path(str(row.get("path") or row.get("data_file"))) for row in rows]


def _extract(paths: list[Path], ncs: dict[str, Any], shot_count: int, batch_samples: int = 4) -> tuple[np.ndarray, np.ndarray, list[str]]:
    torch = require_torch_backend(); bridge = SeismicToVisionBridge(image_size=224, in_chans=3, norm_mode="zscore", feature_mode="raw_envelope_spectrum3"); features, targets = [], []
    for start in range(0, len(paths), batch_samples):
        group = paths[start:start+batch_samples]; images = []
        for path in group:
            sample = load_npz_sample(path); records = torch.as_tensor(sample.records[None, :shot_count], dtype=torch.float32); images.append(bridge(records).detach().cpu().numpy()); targets.append(sample.velocity.astype(np.float32))
        stacked = np.concatenate(images, axis=0); encoded = np.asarray(ncs["model"].encode(stacked), dtype=np.float32).reshape(len(group), shot_count, -1).mean(axis=1); features.extend(encoded)
    return np.asarray(features, dtype=np.float32), np.stack(targets)[:, None].astype(np.float32), _ids(paths)


def _cache(root: Path, family: str, split: str, rows: list[dict[str, Any]], ncs: dict[str, Any], shot_count: int) -> tuple[Path, dict[str, Any]]:
    paths = _paths(rows); ids = _ids(paths); cache = root / f"{family}_{split}_{_hash(ids)[:12]}.npz"; root.mkdir(parents=True, exist_ok=True)
    if not cache.is_file():
        features, target, sample_ids = _extract(paths, ncs, shot_count); np.savez_compressed(cache, features=features, target=target, sample_id=np.asarray(sample_ids, dtype=str), is_real_feature=np.asarray(True), backbone=np.asarray("ncs_2d"), feature_mode=np.asarray("mean_patch"))
    metadata = validate_real_feature_cache(cache)
    with np.load(cache) as payload:
        if payload["sample_id"].astype(str).tolist() != ids: raise ValueError("NCS2D cache sample_id mismatch")
    return cache, metadata


def _load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    validate_real_feature_cache(path)
    with np.load(path) as payload: return payload["features"], payload["target"], payload["sample_id"].astype(str).tolist()


def _train(model: Any, arrays: dict[str, tuple[np.ndarray, np.ndarray, list[str]]], config: dict[str, Any], seed: int, device: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    torch = require_torch_backend(); torch.manual_seed(seed); module = model.module.to(device); first = torch.as_tensor(arrays["train"][0][:1], dtype=torch.float32, device=device)
    with torch.no_grad(): module(first)
    optimizer, parameter_report = build_optimizer_with_registration_report(module, learning_rate=float(config["learning_rate"])); criterion = torch.nn.L1Loss(); history = []; features, target, _ = arrays["train"]
    for epoch in range(1, int(config["epochs"])+1):
        order = np.random.default_rng(seed+epoch).permutation(len(features)); losses=[]; module.train()
        for start in range(0, len(order), int(config["batch_size"])):
            index = order[start:start+int(config["batch_size"])]; x=torch.as_tensor(features[index],dtype=torch.float32,device=device); y=torch.as_tensor(target[index],dtype=torch.float32,device=device); optimizer.zero_grad(set_to_none=True); pred=module(x); loss=criterion(pred,y); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        history.append({"epoch":epoch,"train_l1":float(np.mean(losses))})
    evaluations={}; module.eval()
    with torch.no_grad():
        for split in ("val","in_family_test","cross_family_test"):
            x,y,ids=arrays[split]; parts=[]
            for start in range(0,len(x),int(config["batch_size"])): parts.append(module(torch.as_tensor(x[start:start+int(config["batch_size"])],dtype=torch.float32,device=device)).cpu().numpy())
            pred=np.concatenate(parts).astype(np.float32); metrics=compute_velocity_metrics(pred,y); metrics.update({"metric_space":"physical_velocity","sample_count":len(y),"is_real_feature":True}); evaluations[split]=(metrics,pred,y,ids)
    return history,evaluations,parameter_report


def _write_prediction(path: Path, prediction: np.ndarray, target: np.ndarray, ids: list[str], row: dict[str, Any]) -> None:
    np.savez_compressed(path, velocity_pred_physical=prediction.astype(np.float32), velocity_true_physical=target.astype(np.float32), error_map_physical=np.abs(prediction-target).astype(np.float32), prediction=prediction.astype(np.float32), target=target.astype(np.float32), sample_id=np.asarray(ids,dtype=str), model_id=np.asarray("M6"), bridge_name=np.asarray("raw_envelope_spectrum3"), source_family=np.asarray(row["source_family"]), target_family=np.asarray(row["target_family"]), seed=np.asarray(int(row["seed"])), metric_space=np.asarray("physical_velocity"), is_real_feature=np.asarray(True))


def _complete(run_dir: Path) -> bool:
    if not all((run_dir/name).is_file() for name in REQUIRED): return False
    config=json.loads((run_dir/"config.json").read_text(encoding="utf-8")); card=json.loads((run_dir/"model_card.json").read_text(encoding="utf-8")); metadata=json.loads((run_dir/"feature_cache_metadata.json").read_text(encoding="utf-8")); return config.get("status")=="SUCCESS" and card.get("is_real_feature") is True and card.get("decoder_fully_registered") is True and metadata.get("is_real_feature") is True


def run_ncs2d(*, config_path: str|Path, output_dir: str|Path, seeds: list[int], device: str, resume: bool) -> dict[str, Any]:
    config=yaml.safe_load(Path(config_path).read_text(encoding="utf-8")); root=Path(output_dir); rows=list(csv.DictReader((root/"protocol_v13_run_matrix.csv").open("r",encoding="utf-8",newline=""))); manifests={}
    for path in (root/"manifests").glob("*_manifest.json"):
        data=json.loads(path.read_text(encoding="utf-8"))
        if {"source_family","target_family","seed"}.issubset(data): manifests[(data["source_family"],data["target_family"],int(data["seed"]))]=data
    ncs=load_ncs_model("ncs_2d",repo_path=config["backbones"]["ncs_repo"],weights_path=config["backbones"]["ncs_2d_weights"],device=device)
    if ncs.get("status")!="READY" or not ncs.get("metadata",{}).get("is_real_feature"): raise RuntimeError(f"real NCS2D unavailable: {ncs.get('status')}")
    cache_root=root/"feature_cache"/"ncs_2d"; cache_registry={}; completed=[]
    for row in rows:
        if row["method_key"]!="ncs2d_frozen" or int(row["seed"]) not in seeds: continue
        run_dir=root/"runs"/row["transfer_id"] / "ncs2d_frozen" / f"seed_{row['seed']}"; run_dir.mkdir(parents=True,exist_ok=True)
        if resume and _complete(run_dir): completed.append({**row,"status":"SUCCESS","reused":True}); continue
        start=time.perf_counter(); status="FAILED"; reason=""
        try:
            manifest=manifests[(row["source_family"],row["target_family"],int(row["seed"]))]; split_defs={"train":(row["source_family"],"train",manifest["train_samples"]),"val":(row["source_family"],"val",manifest["val_samples"]),"in_family_test":(row["source_family"],"test",manifest["in_family_test_samples"]),"cross_family_test":(row["target_family"],"test",manifest["cross_family_test_samples"])}; arrays={}; cache_meta={}
            for split,(family,cache_split,samples) in split_defs.items():
                registry_key=(family,cache_split,_hash([str(x.get("sample_id") or x.get("path")) for x in samples]))
                if registry_key not in cache_registry: cache_registry[registry_key]=_cache(cache_root,family,cache_split,samples,ncs,int(config["shot_count"]))
                cache_path,meta=cache_registry[registry_key]; arrays[split]=_load_cache(cache_path); cache_meta[split]=meta
            spec={"method_id":"M6","method_key":"ncs2d_frozen","method_name":"NCS2D frozen","bridge":"raw_envelope_spectrum3","pretraining_source":"seismic_ncs2d","kind":"ncs","adapter":"frozen"}; vmin,vmax=_read_bounds(manifest); model=build_protocol_v11_model(spec,config,vmin=vmin,vmax=vmax); history,evaluations,parameters=_train(model,arrays,config,int(row["seed"]),device)
            _write_history(run_dir/"train_history.csv",history)
            for split,filename in (("val","metrics_val.json"),("in_family_test","metrics_in_family_test.json"),("cross_family_test","metrics_cross_family_test.json")): (run_dir/filename).write_text(json.dumps(evaluations[split][0],indent=2,ensure_ascii=False),encoding="utf-8")
            _write_prediction(run_dir/"predictions_in_family_test.npz",evaluations["in_family_test"][1],evaluations["in_family_test"][2],evaluations["in_family_test"][3],row); _write_prediction(run_dir/"predictions_cross_family_test.npz",evaluations["cross_family_test"][1],evaluations["cross_family_test"][2],evaluations["cross_family_test"][3],row); _write_grid(run_dir/"prediction_grid.png",evaluations["cross_family_test"][1],evaluations["cross_family_test"][2]); _write_grid(run_dir/"gradient_grid.png",evaluations["cross_family_test"][1],evaluations["cross_family_test"][2],gradient=True)
            metadata={"is_real_feature":True,"backbone":"ncs_2d","feature_mode":"mean_patch","shared_cache":True,"splits":cache_meta,"ncs_metadata":ncs.get("metadata",{})}; (run_dir/"feature_cache_metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf-8"); card={**spec,**parameters,"backbone":"ncs_2d","decoder":config["decoder"],"loss":config["loss"],"is_real_feature":True}; (run_dir/"model_card.json").write_text(json.dumps(card,indent=2,ensure_ascii=False),encoding="utf-8"); status="SUCCESS"
        except Exception as exc:
            reason=f"{type(exc).__name__}: {exc}"; (run_dir/"exception.txt").write_text(traceback.format_exc(),encoding="utf-8")
        run_config={**row,"status":status,"skip_reason":reason,"runtime_seconds":time.perf_counter()-start,"device":device,"is_real_feature":status=="SUCCESS","manifest_combined_hash":manifests.get((row["source_family"],row["target_family"],int(row["seed"])),{}).get("manifest_combined_hash"),"target_test_used_for_training":False,"target_test_used_for_validation":False,"target_test_used_for_model_selection":False}; config_hash=_hash({"config":config,"run":{k:v for k,v in row.items() if k not in {"status"}}}); run_config["locked_config_hash"]=config_hash; (run_dir/"config.json").write_text(json.dumps(run_config,indent=2,ensure_ascii=False),encoding="utf-8"); (run_dir/"config_hash.txt").write_text(config_hash+"\n",encoding="utf-8"); (run_dir/"run_log.txt").write_text(f"status={status}\nreason={reason}\nruntime_seconds={run_config['runtime_seconds']:.3f}\n",encoding="utf-8"); completed.append(run_config)
    summary={"run_count":len(completed),"success":sum(row["status"]=="SUCCESS" for row in completed),"failed":sum(row["status"]=="FAILED" for row in completed),"runs":completed}; (root/"protocol_v13_ncs2d_run_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8"); return summary


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);parser.add_argument("--output-dir",required=True);parser.add_argument("--seeds",type=int,nargs="+",required=True);parser.add_argument("--device",default="cpu");parser.add_argument("--resume",default="true");args=parser.parse_args();result=run_ncs2d(config_path=args.config,output_dir=args.output_dir,seeds=args.seeds,device=args.device,resume=_bool(args.resume));print(json.dumps({k:result[k] for k in ("run_count","success","failed")},indent=2))


if __name__=="__main__":main()
