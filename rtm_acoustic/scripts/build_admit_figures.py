from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = [
    "initial",
    "full_fwi",
    "global",
    "illumination",
    "consensus",
    "depth",
    "inverse",
    "ecg",
    "random_seed_0",
    "random_seed_1",
    "random_seed_2",
    "random_seed_3",
    "random_seed_4",
]
SELECTED_METHODS = ["initial", "full_fwi", "global", "illumination", "ecg", "random_seed_4", "inverse"]
ROI_REGIONS = ["salt_top", "salt_flanks", "subsalt_shadow", "low_illumination", "deep_roi"]


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def _save(fig: plt.Figure, path: Path, paper_figures: Path | None = None) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in [".png", ".pdf"]:
        out = path.with_suffix(suffix)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        outputs.append(out)
        if paper_figures is not None:
            paper_figures.mkdir(parents=True, exist_ok=True)
            shutil.copy2(out, paper_figures / out.name)
    plt.close(fig)
    return outputs


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    order = {name: i for i, name in enumerate(METHOD_ORDER)}
    return df.assign(_order=df["method"].map(order).fillna(999)).sort_values("_order").drop(columns="_order")


def plot_split_bars(split_csv: Path, out_dir: Path, paper_figures: Path) -> None:
    df = _ordered(pd.read_csv(split_csv))
    df = df[df["status"] == "READY"].copy()
    x = np.arange(len(df))
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), constrained_layout=True)
    color = "#0072B2"
    axes[0].bar(x, df["rtm_split_laplacian_correlation"], color=color, edgecolor="0.2", linewidth=0.4)
    axes[0].axhline(float(df.loc[df["method"] == "initial", "rtm_split_laplacian_correlation"].iloc[0]), color="0.25", linestyle="--", linewidth=0.8, label="initial")
    axes[0].set_ylabel("Laplacian split corr.")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([])
    axes[0].legend(frameon=False)
    axes[1].bar(x, df["local_structure_tensor_coherence"], color="#009E73", edgecolor="0.2", linewidth=0.4)
    axes[1].set_ylabel("Structure coherence")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["method"], rotation=45, ha="right")
    for ax, label in zip(axes, ["a", "b"]):
        ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontweight="bold", va="top")
        ax.grid(axis="y", color="0.88", linewidth=0.5)
    _save(fig, out_dir / "figure_split_metric_bars", paper_figures)


def plot_pairwise_delta(pair_csv: Path, out_dir: Path, paper_figures: Path) -> None:
    df = _ordered(pd.read_csv(pair_csv))
    df = df[df["status"] == "READY"].copy()
    colors = np.where(df["delta_laplacian_split_corr"] >= 0, "#0072B2", "#D55E00")
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    ax.bar(np.arange(len(df)), df["delta_laplacian_split_corr"], color=colors, edgecolor="0.25", linewidth=0.4)
    ax.axhline(0.0, color="0.2", linewidth=0.8)
    ax.set_ylabel("Delta vs illumination")
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels(df["method"], rotation=45, ha="right")
    ax.set_title("Split consistency relative to illumination-only")
    ax.grid(axis="y", color="0.88", linewidth=0.5)
    _save(fig, out_dir / "figure_pairwise_split_delta", paper_figures)


def _load_image(root: Path, subset: str, method: str) -> np.ndarray:
    return np.load(root / subset / method / "rtm_laplacian_filtered_physical.npy")


def plot_split_images(split_rtm_root: Path, out_dir: Path, paper_figures: Path) -> None:
    methods = ["initial", "full_fwi", "illumination", "ecg", "random_seed_4"]
    images = [_load_image(split_rtm_root, subset, method) for subset in ["subset_A", "subset_B"] for method in methods]
    clip = float(np.percentile(np.abs(np.concatenate([img.ravel() for img in images])), 99.5))
    clip = clip if clip > 0 else 1.0
    fig, axes = plt.subplots(2, len(methods), figsize=(7.2, 3.2), constrained_layout=True)
    for row, subset in enumerate(["subset_A", "subset_B"]):
        for col, method in enumerate(methods):
            ax = axes[row, col]
            ax.imshow(_load_image(split_rtm_root, subset, method), cmap="RdBu_r", vmin=-clip, vmax=clip, aspect="auto")
            if row == 0:
                ax.set_title(method)
            if col == 0:
                ax.set_ylabel(subset)
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].text(-0.18, 1.08, "a", transform=axes[0, 0].transAxes, fontweight="bold")
    axes[1, 0].text(-0.18, 1.08, "b", transform=axes[1, 0].transAxes, fontweight="bold")
    _save(fig, out_dir / "figure_split_rtm_images", paper_figures)


def plot_roi_figures(roi_csv: Path, out_dir: Path, paper_figures: Path) -> None:
    df = pd.read_csv(roi_csv)
    df = df[df["method"].isin(SELECTED_METHODS) & df["region"].isin(ROI_REGIONS)].copy()
    df["method"] = pd.Categorical(df["method"], SELECTED_METHODS, ordered=True)
    df["region"] = pd.Categorical(df["region"], ROI_REGIONS, ordered=True)

    def heatmap(value: str, name: str, title: str, log: bool = False, cmap: str = "viridis") -> None:
        pivot = df.pivot(index="region", columns="method", values=value).astype(float)
        data = np.log10(pivot + 1.0e-30) if log else pivot
        fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label(f"log10({value})" if log else value)
        _save(fig, out_dir / name, paper_figures)

    heatmap("rtm_laplacian_energy", "figure_roi_rtm_energy", "ROI RTM Laplacian Energy", log=True, cmap="viridis")
    heatmap("update_l2", "figure_roi_update_energy", "ROI Update L2", log=True, cmap="magma")
    heatmap("rtm_split_laplacian_correlation", "figure_roi_admissibility_matrix", "ROI-Admissibility Matrix", log=False, cmap="cividis")


def plot_evidence_figures(evidence_csv: Path, out_dir: Path, paper_figures: Path) -> None:
    df = _ordered(pd.read_csv(evidence_csv))
    metrics = [
        "data_space_nrms",
        "model_space_rmse",
        "image_space_filtered_rmse",
        "split_consistency_corr",
    ]
    data = df.set_index("method")[metrics].apply(pd.to_numeric, errors="coerce")
    normalized = data.copy()
    for col in normalized:
        values = normalized[col]
        lo, hi = float(values.min()), float(values.max())
        normalized[col] = (values - lo) / (hi - lo) if hi > lo else 0.0
    fig, ax = plt.subplots(figsize=(7.2, 4.1), constrained_layout=True)
    im = ax.imshow(normalized, cmap="cividis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(["Data NRMS", "Model RMSE", "RTM RMSE", "Split corr."], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(normalized.index)))
    ax.set_yticklabels(normalized.index)
    ax.set_title("Data-model-image-split evidence matrix")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Normalized value")
    _save(fig, out_dir / "figure_data_model_image_deeptime_matrix", paper_figures)

    rank_df = df[["method", "data_space_nrms", "image_space_filtered_rmse", "split_consistency_corr"]].copy()
    rank_df["data_rank"] = rank_df["data_space_nrms"].rank(method="min", ascending=True)
    rank_df["image_rank"] = rank_df["image_space_filtered_rmse"].rank(method="min", ascending=True)
    rank_df["split_rank"] = rank_df["split_consistency_corr"].rank(method="min", ascending=False)
    ranks = rank_df.set_index("method")[["data_rank", "image_rank", "split_rank"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    im = ax.imshow(ranks, cmap="viridis_r", aspect="auto")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Data", "Image", "Split"])
    ax.set_yticks(np.arange(len(ranks.index)))
    ax.set_yticklabels(ranks.index)
    ax.set_title("Method ranking by audit domain")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Rank (lower is better)")
    _save(fig, out_dir / "figure_method_ranking_by_domain", paper_figures)


def write_stage_report(root: Path, figure_paths: list[Path]) -> Path:
    split = pd.read_csv(root / "outputs/admit_fwi_v1/seg_salt_main_case/split_consistency/split_metrics.csv")
    evidence = pd.read_csv(root / "outputs/admit_fwi_v1/seg_salt_main_case/evidence_matrix/admit_evidence_matrix.csv")
    ecg = split.loc[split["method"] == "ecg", "rtm_split_laplacian_correlation"].iloc[0]
    illumination = split.loc[split["method"] == "illumination", "rtm_split_laplacian_correlation"].iloc[0]
    random4 = split.loc[split["method"] == "random_seed_4", "rtm_split_laplacian_correlation"].iloc[0]
    verdict_rows = evidence[["method", "overall_admissibility_verdict", "deep_time_status"]].astype(str).to_dict("records")
    verdict_table = [
        "| method | overall_admissibility_verdict | deep_time_status |",
        "|---|---|---|",
    ]
    verdict_table.extend(
        f"| {row['method']} | {row['overall_admissibility_verdict']} | {row['deep_time_status']} |"
        for row in verdict_rows
    )
    lines = [
        "# ADMIT-FWI split/ROI/evidence 图件阶段进展报告",
        "",
        "生成日期：2026-07-08",
        "",
        "## 完成情况",
        "",
        "- 已基于真实 `subset_A/subset_B` RTM 输出生成 split consistency 图件。",
        "- 已基于 ROI diagnostics 生成 ROI 能量、更新量和 admissibility matrix 图件。",
        "- 已基于 ADMIT evidence matrix 生成跨域证据热图和方法排名图。",
        "- 图件已同步复制到 `outputs/admit_fwi_v1/paper_assets/figures/`。",
        "",
        "## 关键 split 指标",
        "",
        f"- illumination `rtm_split_laplacian_correlation`: `{illumination:.6g}`",
        f"- ECG `rtm_split_laplacian_correlation`: `{ecg:.6g}`",
        f"- random_seed_4 `rtm_split_laplacian_correlation`: `{random4:.6g}`",
        "",
        "当前 split 指标不支持“ECG 显著优于 illumination-only”。",
        "",
        "## Evidence verdict 摘要",
        "",
        "\n".join(verdict_table),
        "",
        "## 生成图件",
        "",
    ]
    lines.extend(f"- `{path}`" for path in figure_paths if path.suffix == ".png")
    lines.extend(
        [
            "",
            "## 可写入论文的结论",
            "",
            "- ADMIT-FWI now includes true split-RTM image consistency for the SEG/Salt short-record case.",
            "- Spatial selective gates can be audited against global, inverse, and random controls.",
            "- Illumination-only remains a strong baseline.",
            "- ECG is an evidence-calibrated candidate, but current split/ROI metrics do not establish unique superiority.",
            "",
            "## 仍禁止的结论",
            "",
            "- ECG significantly improves FWI/RTM imaging.",
            "- ADMIT-FWI solves subsalt velocity building.",
            "- Short-record split RTM proves deep imaging quality.",
            "- Full FWI is most reliable only because residual is lowest.",
        ]
    )
    out = root / "outputs/admit_fwi_v1/stage_progress_report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    docs_out = root / "docs/admit_fwi_stage_progress_report.md"
    docs_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ADMIT-FWI split/ROI/evidence publication figures.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root
    _setup_style()
    paper_figures = root / "outputs/admit_fwi_v1/paper_assets/figures"
    split_dir = root / "outputs/admit_fwi_v1/seg_salt_main_case/split_consistency"
    roi_dir = root / "outputs/admit_fwi_v1/seg_salt_main_case/roi_diagnostics"
    evidence_dir = root / "outputs/admit_fwi_v1/seg_salt_main_case/evidence_matrix"

    plot_split_bars(split_dir / "split_metrics.csv", split_dir / "figures", paper_figures)
    plot_pairwise_delta(split_dir / "split_pairwise_vs_illumination.csv", split_dir / "figures", paper_figures)
    plot_split_images(root / "outputs/RTM/audit0_gate_rtm_split_v1", split_dir / "figures", paper_figures)
    plot_roi_figures(roi_dir / "roi_metrics.csv", roi_dir / "figures", paper_figures)
    plot_evidence_figures(evidence_dir / "admit_evidence_matrix.csv", evidence_dir / "figures", paper_figures)

    figures = list(split_dir.glob("figures/*.png")) + list(roi_dir.glob("figures/*.png")) + list(evidence_dir.glob("figures/*.png"))
    report = write_stage_report(root, figures)
    print(f"wrote {len(figures)} PNG figures and report {report}")


if __name__ == "__main__":
    main()
