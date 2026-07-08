# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from fwi_visionfm.scripts.build_protocol_v2_splits import discover_families


CSV_FIELDS = ["sample_id", "family", "data_file", "model_file", "local_index", "global_index", "path"]


def _read_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["sample_id"] for row in csv.DictReader(handle)}


def validate_locked_splits(manifest_dir: str | Path, families: list[str]) -> None:
    root = Path(manifest_dir)
    for family in families:
        paths = [root / f"{family}_train200.csv", root / f"{family}_val50.csv", root / f"{family}_test50.csv"]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"locked split missing for {family}")
        train, val, test = (_read_ids(path) for path in paths)
        if train & val or train & test or val & test:
            raise ValueError(f"locked split overlap detected for {family}")


def compute_manifest_hashes(manifest_dir: str | Path) -> dict[str, str]:
    root = Path(manifest_dir)
    result: dict[str, str] = {}
    for path in sorted(root.glob("*_*.csv")):
        result[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def combined_manifest_hash(hashes: dict[str, str]) -> str:
    canonical = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _family_rows(discovered: dict[str, list[dict[str, Any]]], directory_name: str) -> list[dict[str, Any]]:
    for name, rows in discovered.items():
        if name.lower() == directory_name.lower():
            return rows
    raise ValueError(f"family data unavailable: {directory_name}")


def _sample_id(row: dict[str, Any]) -> str:
    path = Path(str(row["path"]))
    if path.suffix.lower() == ".npz":
        return f"{path.parent.name}/{path.name}"
    return f"{row['family']}/{path.name}:{int(row['local_index'])}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({"sample_id": _sample_id(row), **{field: row[field] for field in CSV_FIELDS if field != "sample_id"}})


def build_locked_manifests(*, config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    discovered = discover_families(config["data_root"])
    total = int(config["train_size"]) + int(config["val_size"]) + int(config["in_family_test_size"])
    rng = np.random.default_rng(int(config.get("manifest_seed", 1200)))
    family_keys = list(config["families"])
    for family_key, directory_name in config["families"].items():
        rows = _family_rows(discovered, directory_name)
        if len(rows) < total:
            raise ValueError(f"{family_key} has {len(rows)} samples, requires {total}")
        selected = [rows[int(index)] for index in rng.permutation(len(rows))[:total]]
        start_val = int(config["train_size"]); start_test = start_val + int(config["val_size"])
        _write_csv(out / f"{family_key}_train200.csv", selected[:start_val])
        _write_csv(out / f"{family_key}_val50.csv", selected[start_val:start_test])
        _write_csv(out / f"{family_key}_test50.csv", selected[start_test:])
    validate_locked_splits(out, family_keys)
    hashes = compute_manifest_hashes(out)
    payload = {"protocol": config["protocol"], "hash_algorithm": "sha256", "files": hashes, "combined_hash": combined_manifest_hash(hashes)}
    (out / "protocol_v12_manifest_hashes.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Protocol V12 锁定数据清单报告", "", f"- family 数：{len(family_keys)}", f"- 合并 SHA256：`{payload['combined_hash']}`", "- 每个 family 固定使用 200 train / 50 val / 50 test。", "- train、val、test sample_id 已验证互不重叠。", "- 所有方法和 seed 复用相同清单。", ""]
    (out / "protocol_v12_manifest_report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--repo-root", required=True); parser.add_argument("--config", required=True); parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(); repo = Path(args.repo_root); config = Path(args.config) if Path(args.config).is_absolute() else repo / args.config; output = Path(args.output_dir) if Path(args.output_dir).is_absolute() else repo / args.output_dir
    print(json.dumps(build_locked_manifests(config_path=config, output_dir=output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

