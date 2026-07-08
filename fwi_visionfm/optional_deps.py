from __future__ import annotations

import importlib.util

OPTIONAL_DEPENDENCIES = ("torch", "torchvision", "timm", "transformers", "segment_anything")


def check_optional_dependencies() -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    for name in OPTIONAL_DEPENDENCIES:
        spec = importlib.util.find_spec(name)
        report[name] = {"available": spec is not None, "module": name}
    return report


def missing_dependencies(*names: str) -> list[str]:
    report = check_optional_dependencies()
    return [name for name in names if not bool(report.get(name, {}).get("available", False))]
