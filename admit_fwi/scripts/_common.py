from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


def read_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- ") and current_list_key:
            data.setdefault(current_list_key, []).append(line[2:].strip())
            continue
        current_list_key = None
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not value:
            data[key] = []
            current_list_key = key
        else:
            try:
                data[key] = ast.literal_eval(value)
            except Exception:
                data[key] = value
    return data


def ensure_output_tree(output_dir: Path) -> None:
    for child in ["diagnostics", "gates", "models", "audit", "rtm", "tables", "figures", "report"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

