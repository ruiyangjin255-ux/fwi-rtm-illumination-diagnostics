from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _load_rows(root: Path) -> list[dict[str, str]]:
    summary_path = root / "protocol_v7_boundary_auxiliary_tuning_summary.csv"
    with summary_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _success_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("status") == "SUCCESS"]


def _group_lambda(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = {}
    for row in _success_rows(rows):
        if row["boundary_method"] == "gradient_magnitude" and row["lambda_boundary"]:
            groups.setdefault(row["lambda_boundary"], []).append(row)
    return groups


def _best_lambda(groups: dict[str, list[dict[str, str]]]) -> tuple[str, str]:
    scored = []
    for lambda_value, rows in groups.items():
        if not rows:
            continue
        grad = sum(float(row["cross_family_gradient_error"]) for row in rows) / len(rows)
        edge = sum(float(row["cross_family_edge_MAE"]) for row in rows) / len(rows)
        mae = sum(float(row["cross_family_MAE"]) for row in rows) / len(rows)
        rmse = sum(float(row["cross_family_RMSE"]) for row in rows) / len(rows)
        scored.append((lambda_value, grad, edge, mae, rmse))
    scored.sort(key=lambda item: (item[1], item[2], item[3], item[4]))
    if not scored:
        return "", ""
    best = scored[0]
    return best[0], f"structural metrics mean lower with limited MAE/RMSE trade-off at lambda={best[0]}"


def _best_method_seed0(rows: list[dict[str, str]]) -> tuple[str, str]:
    if not rows:
        return "", ""
    ranked = sorted(
        rows,
        key=lambda row: (
            float(row["cross_family_gradient_error"]),
            float(row["cross_family_edge_MAE"]),
            float(row["cross_family_MAE"]),
            float(row["cross_family_RMSE"]),
        ),
    )
    best = ranked[0]
    return best["boundary_method"], (
        f"seed=0 only comparison favors {best['boundary_method']} "
        f"(gradient_error={best['cross_family_gradient_error']}, edge_MAE={best['cross_family_edge_MAE']})"
    )


def write_protocol_v7_boundary_auxiliary_tuning_report(root: str | Path, reuse_seed_stability_root: str | Path | None = None) -> Path:
    del reuse_seed_stability_root
    root_path = Path(root)
    rows = _load_rows(root_path)
    lambda_groups = _group_lambda(rows)
    recommended_lambda, recommended_reason = _best_lambda(lambda_groups)
    method_rows = [row for row in _success_rows(rows) if row["seed"] == "0" and row["lambda_boundary"] == "0.05" and row["boundary_method"]]
    best_method, best_method_reason = _best_method_seed0(method_rows)

    lines = [
        "# Protocol V7 Boundary Auxiliary Tuning Report",
        "",
        "## 1. Goal",
        "本轮只围绕 selected boundary auxiliary 做小范围 lambda / boundary target 调参，不新增 backbone，不做 benchmark claim。",
        "",
        "## 2. Reused Results",
        "- baseline seed=0/1/2",
        "- lambda=0.10 + gradient_magnitude seed=0/1/2",
    ]
    reused_smoke = [row for row in rows if row.get("reused_from", "").startswith("smoke:")]
    for row in reused_smoke:
        lines.append(f"- {row['run_id']} reused from {row['reused_from']}")
    lines.extend(
        [
            "",
            "## 3. Lambda Tuning",
            "| lambda_boundary | seed | MAE | RMSE | SSIM | gradient_error | edge_MAE |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for lambda_value in sorted(lambda_groups.keys(), key=float):
        for row in sorted(lambda_groups[lambda_value], key=lambda item: int(item["seed"])):
            lines.append(
                f"| {lambda_value} | {row['seed']} | {row['cross_family_MAE']} | {row['cross_family_RMSE']} | {row['cross_family_SSIM']} | {row['cross_family_gradient_error']} | {row['cross_family_edge_MAE']} |"
            )
    lines.extend(
        [
            "- 重点判断：哪个 lambda 的 gradient_error / edge_MAE 最稳、哪个 lambda 的 MAE/RMSE trade-off 最小、SSIM 是否仍不稳定。",
            "",
            "## 4. Boundary Method Tuning",
            "| boundary_method | seed | lambda_boundary | threshold | MAE | RMSE | SSIM | gradient_error | edge_MAE |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in sorted(method_rows, key=lambda item: item["boundary_method"]):
        lines.append(
            f"| {row['boundary_method']} | {row['seed']} | {row['lambda_boundary']} | {row['threshold']} | {row['cross_family_MAE']} | {row['cross_family_RMSE']} | {row['cross_family_SSIM']} | {row['cross_family_gradient_error']} | {row['cross_family_edge_MAE']} |"
        )
    lines.extend(
        [
            "- 如果 sobel / thresholded_gradient 没有明显优势，不扩大到多 seed。",
            f"- method readout: {best_method_reason or 'method comparison unavailable'}。",
            "",
            "## 5. Recommended Setting",
            f"- recommended selected setting: boundary_method=gradient_magnitude, lambda_boundary={recommended_lambda or '0.10'}",
            f"- reason: {recommended_reason or 'reuse result remains the most complete and stable'}",
            f"- method note: {best_method or 'gradient_magnitude'} is not upgraded to the final recommendation until seed=1/2 are checked under the same lambda.",
            "",
            "## 6. Limitations",
            "- CPU small-sample",
            "- train_size=100 / val_size=50 / test_size=50",
            "- epochs=2 only",
            "- selected tuning only",
            "- not benchmark-level proof",
            "- no DINOv2/SAM/NCS",
            "- boundary targets are derived from velocity gradients, not manually labeled geology",
            "",
            "## 7. Next Step",
            "- 如果某个 lambda 在 gradient_error 和 edge_MAE 上保持 3/3 且 MAE/RMSE 不恶化，可进入 V7 selected final report。",
            "- 如果 lambda 0.03/0.05 降低结构收益，保留 lambda=0.10。",
            "- 如果 sobel 或 thresholded_gradient 在 seed=0 明显优于 gradient_magnitude，后续只对该 method 做 seed=1/2，不扩大其他矩阵。",
            "",
            "允许写法：",
            "- Boundary auxiliary tuning suggests which selected configuration is more stable under CPU small-sample settings.",
            "- Results remain selected tuning evidence rather than benchmark-level proof.",
        ]
    )
    report_path = root_path / "protocol_v7_boundary_auxiliary_tuning_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write Protocol V7 boundary auxiliary tuning report.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reuse-seed-stability-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(write_protocol_v7_boundary_auxiliary_tuning_report(args.root, reuse_seed_stability_root=args.reuse_seed_stability_root))


if __name__ == "__main__":
    main()
