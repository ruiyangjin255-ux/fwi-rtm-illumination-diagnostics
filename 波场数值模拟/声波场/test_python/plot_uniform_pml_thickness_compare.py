from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figure"

DOMAIN_M = 4000.0
DX = 10.0
DZ = 10.0
DT = 0.001
T_SNAP = 0.80
F0 = 30.0
V0 = 3200.0
ORDER = 4
PML_THICKNESSES = [10, 20, 30]
CJK_FONT = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf")


def staggered_coefficients(order: int) -> np.ndarray:
    if order == 2:
        return np.array([1.0], dtype=np.float64)
    if order == 4:
        return np.array([9.0 / 8.0, -1.0 / 24.0], dtype=np.float64)
    if order == 6:
        return np.array([75.0 / 64.0, -25.0 / 384.0, 3.0 / 640.0], dtype=np.float64)
    raise ValueError("order must be 2, 4, or 6")


def derivative_x_forward(a: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    d = np.zeros_like(a)
    n = len(coeff)
    idx = np.arange(n, a.shape[1] - n)
    cols = slice(n, a.shape[1] - n)
    for m, c in enumerate(coeff, start=1):
        d[:, cols] += c * (a[:, idx + m] - a[:, idx - m + 1])
    return d / DX


def derivative_x_backward(a: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    d = np.zeros_like(a)
    n = len(coeff)
    idx = np.arange(n, a.shape[1] - n)
    cols = slice(n, a.shape[1] - n)
    for m, c in enumerate(coeff, start=1):
        d[:, cols] += c * (a[:, idx + m - 1] - a[:, idx - m])
    return d / DX


def derivative_z_forward(a: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    d = np.zeros_like(a)
    n = len(coeff)
    idx = np.arange(n, a.shape[0] - n)
    rows = slice(n, a.shape[0] - n)
    for m, c in enumerate(coeff, start=1):
        d[rows, :] += c * (a[idx + m, :] - a[idx - m + 1, :])
    return d / DZ


def derivative_z_backward(a: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    d = np.zeros_like(a)
    n = len(coeff)
    idx = np.arange(n, a.shape[0] - n)
    rows = slice(n, a.shape[0] - n)
    for m, c in enumerate(coeff, start=1):
        d[rows, :] += c * (a[idx + m - 1, :] - a[idx - m, :])
    return d / DZ


def ricker_wavelet(nt: int) -> np.ndarray:
    t = np.arange(nt) * DT
    t0 = 1.0 / F0
    arg = (np.pi * F0 * (t - t0)) ** 2
    return (1.0 - 2.0 * arg) * np.exp(-arg)


def pml_damping(nz: int, nx: int, pml: int) -> np.ndarray:
    sigma = np.zeros((nz, nx), dtype=np.float64)
    sigma_max = 45.0
    order = 2

    for i in range(pml):
        value = sigma_max * ((pml - i) / pml) ** order
        sigma[:, i] += value
        sigma[:, -i - 1] += value
        sigma[i, :] += value
        sigma[-i - 1, :] += value

    return np.exp(-sigma * DT)


def simulate(pml: int) -> np.ndarray:
    nx0 = int(round(DOMAIN_M / DX)) + 1
    nz0 = int(round(DOMAIN_M / DZ)) + 1
    nx = nx0 + 2 * pml
    nz = nz0 + 2 * pml
    nt = int(round(T_SNAP / DT))
    coeff = staggered_coefficients(ORDER)
    wavelet = ricker_wavelet(nt)
    damp = pml_damping(nz, nx, pml)

    px = np.zeros((nz, nx), dtype=np.float64)
    pz = np.zeros((nz, nx), dtype=np.float64)
    vx = np.zeros((nz, nx), dtype=np.float64)
    vz = np.zeros((nz, nx), dtype=np.float64)

    src_z = nz // 2
    src_x = nx // 2
    kappa = V0 * V0 * DT

    for it in range(nt):
        p = px + pz
        vx += DT * derivative_x_forward(p, coeff)
        vz += DT * derivative_z_forward(p, coeff)
        vx *= damp
        vz *= damp

        px += kappa * derivative_x_backward(vx, coeff)
        pz += kappa * derivative_z_backward(vz, coeff)

        px[src_z, src_x] += wavelet[it]
        pz[src_z, src_x] += wavelet[it]
        px *= damp
        pz *= damp

    return (px + pz)[pml:pml + nz0, pml:pml + nx0]


def add_right_cax(ax):
    divider = make_axes_locatable(ax)
    return divider.append_axes("right", size="3%", pad=0.04)


def setup_axis(ax):
    ax.set_xlabel("Distance (km)", fontsize=10, labelpad=3)
    ax.set_ylabel("Depth (km)", fontsize=10, labelpad=3)
    ax.xaxis.set_major_locator(MultipleLocator(0.8))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="major", labelsize=9, width=1.0)
    ax.tick_params(which="minor", width=0.7)


def plot_compare():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = []
    for pml in PML_THICKNESSES:
        print(f"Simulating PML thickness {pml} grid points...")
        fields.append(simulate(pml))

    scale = max(np.max(np.abs(field)) for field in fields)
    fields = [field / scale for field in fields]

    x_max = DOMAIN_M / 1000.0
    z_max = DOMAIN_M / 1000.0
    extent = [0.0, x_max, z_max, 0.0]

    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.95), facecolor="white")
    labels = [
        "a.PML厚度10个网格点",
        "b.PML厚度20个网格点",
        "c.PML厚度30个网格点",
    ]

    for ax, field, label in zip(axes, fields, labels):
        im = ax.imshow(
            field,
            cmap="seismic",
            vmin=-0.50,
            vmax=0.50,
            extent=extent,
            origin="upper",
            aspect="equal",
        )
        setup_axis(ax)
        ax.text(
            0.5,
            -0.21,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9.4,
            fontweight="bold",
            fontproperties=CJK_FONT,
        )
        cbar = fig.colorbar(im, cax=add_right_cax(ax))
        cbar.set_label("Amplitude", fontsize=9)
        cbar.ax.tick_params(labelsize=8.5)

    fig.subplots_adjust(wspace=0.34, bottom=0.19)
    png = OUT_DIR / "uniform_pml_thickness_wavefield_0p8s.png"
    pdf = OUT_DIR / "uniform_pml_thickness_wavefield_0p8s.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    plot_compare()
