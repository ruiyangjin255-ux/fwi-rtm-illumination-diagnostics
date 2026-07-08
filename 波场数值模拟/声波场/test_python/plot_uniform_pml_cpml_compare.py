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
T_SNAP = 0.95
F0 = 22.0
V0 = 2800.0
ORDER = 4
PML_CELLS = 20
SRC_X_M = 2000.0
SRC_Z_M = 2000.0
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


def damping_profile_1d(n: int, pml: int, sigma_max: float, power: int = 2) -> np.ndarray:
    sigma = np.zeros(n, dtype=np.float64)
    for i in range(pml):
        value = sigma_max * ((pml - i) / pml) ** power
        sigma[i] = value
        sigma[-i - 1] = value
    return sigma


def pml_damping(nz: int, nx: int, pml: int) -> np.ndarray:
    sigma_x = damping_profile_1d(nx, pml, sigma_max=12.0)
    sigma_z = damping_profile_1d(nz, pml, sigma_max=12.0)
    sigma = sigma_z[:, None] + sigma_x[None, :]
    return np.exp(-sigma * DT)


def cpml_profiles_1d(
    n: int,
    pml: int,
    sigma_max: float = 32.0,
    kappa_max: float = 2.2,
    alpha_max: float = 2.0 * np.pi * F0 * 0.02,
    power: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma = np.zeros(n, dtype=np.float64)
    kappa = np.ones(n, dtype=np.float64)
    alpha = np.zeros(n, dtype=np.float64)

    for i in range(pml):
        distance = (pml - i) / pml
        sigma_value = sigma_max * distance ** power
        kappa_value = 1.0 + (kappa_max - 1.0) * distance ** power
        alpha_value = alpha_max * (1.0 - distance)

        sigma[i] = sigma_value
        sigma[-i - 1] = sigma_value
        kappa[i] = kappa_value
        kappa[-i - 1] = kappa_value
        alpha[i] = alpha_value
        alpha[-i - 1] = alpha_value

    b = np.exp(-(sigma / kappa + alpha) * DT)
    a = np.zeros(n, dtype=np.float64)
    denom = kappa * (sigma + kappa * alpha)
    mask = denom > 0.0
    a[mask] = sigma[mask] * (b[mask] - 1.0) / denom[mask]
    return a, b, kappa


def crop_effective(field: np.ndarray) -> np.ndarray:
    nx0 = int(round(DOMAIN_M / DX)) + 1
    nz0 = int(round(DOMAIN_M / DZ)) + 1
    return field[PML_CELLS:PML_CELLS + nz0, PML_CELLS:PML_CELLS + nx0]


def simulate_pml() -> np.ndarray:
    nx0 = int(round(DOMAIN_M / DX)) + 1
    nz0 = int(round(DOMAIN_M / DZ)) + 1
    nx = nx0 + 2 * PML_CELLS
    nz = nz0 + 2 * PML_CELLS
    nt = int(round(T_SNAP / DT))
    coeff = staggered_coefficients(ORDER)
    wavelet = ricker_wavelet(nt)
    damp = pml_damping(nz, nx, PML_CELLS)

    px = np.zeros((nz, nx), dtype=np.float64)
    pz = np.zeros((nz, nx), dtype=np.float64)
    vx = np.zeros((nz, nx), dtype=np.float64)
    vz = np.zeros((nz, nx), dtype=np.float64)
    src_x = PML_CELLS + int(round(SRC_X_M / DX))
    src_z = PML_CELLS + int(round(SRC_Z_M / DZ))
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

    return crop_effective(px + pz)


def cpml_x(derivative: np.ndarray, psi: np.ndarray, a: np.ndarray, b: np.ndarray, kappa: np.ndarray):
    psi = b[None, :] * psi + a[None, :] * derivative
    return derivative / kappa[None, :] + psi, psi


def cpml_z(derivative: np.ndarray, psi: np.ndarray, a: np.ndarray, b: np.ndarray, kappa: np.ndarray):
    psi = b[:, None] * psi + a[:, None] * derivative
    return derivative / kappa[:, None] + psi, psi


def simulate_cpml() -> np.ndarray:
    nx0 = int(round(DOMAIN_M / DX)) + 1
    nz0 = int(round(DOMAIN_M / DZ)) + 1
    nx = nx0 + 2 * PML_CELLS
    nz = nz0 + 2 * PML_CELLS
    nt = int(round(T_SNAP / DT))
    coeff = staggered_coefficients(ORDER)
    wavelet = ricker_wavelet(nt)

    ax, bx, kx = cpml_profiles_1d(nx, PML_CELLS)
    az, bz, kz = cpml_profiles_1d(nz, PML_CELLS)

    px = np.zeros((nz, nx), dtype=np.float64)
    pz = np.zeros((nz, nx), dtype=np.float64)
    vx = np.zeros((nz, nx), dtype=np.float64)
    vz = np.zeros((nz, nx), dtype=np.float64)
    psi_vx = np.zeros((nz, nx), dtype=np.float64)
    psi_vz = np.zeros((nz, nx), dtype=np.float64)
    psi_px = np.zeros((nz, nx), dtype=np.float64)
    psi_pz = np.zeros((nz, nx), dtype=np.float64)

    src_x = PML_CELLS + int(round(SRC_X_M / DX))
    src_z = PML_CELLS + int(round(SRC_Z_M / DZ))
    kappa = V0 * V0 * DT

    for it in range(nt):
        p = px + pz
        dpdx, psi_vx = cpml_x(derivative_x_forward(p, coeff), psi_vx, ax, bx, kx)
        dpdz, psi_vz = cpml_z(derivative_z_forward(p, coeff), psi_vz, az, bz, kz)
        vx += DT * dpdx
        vz += DT * dpdz

        dvxdx, psi_px = cpml_x(derivative_x_backward(vx, coeff), psi_px, ax, bx, kx)
        dvzdz, psi_pz = cpml_z(derivative_z_backward(vz, coeff), psi_pz, az, bz, kz)
        px += kappa * dvxdx
        pz += kappa * dvzdz

        px[src_z, src_x] += wavelet[it]
        pz[src_z, src_x] += wavelet[it]

    return crop_effective(px + pz)


def add_right_cax(ax):
    divider = make_axes_locatable(ax)
    return divider.append_axes("right", size="3%", pad=0.04)


def setup_axis(ax):
    ax.set_xlabel("Distance (km)", fontsize=11, labelpad=4)
    ax.set_ylabel("Depth (km)", fontsize=11, labelpad=4)
    ax.xaxis.set_major_locator(MultipleLocator(0.8))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="major", labelsize=10, width=1.1)
    ax.tick_params(which="minor", width=0.8)


def plot_compare():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Simulating PML...")
    pml = simulate_pml()
    print("Simulating CPML...")
    cpml = simulate_cpml()

    scale = max(np.max(np.abs(pml)), np.max(np.abs(cpml)))
    pml = pml / scale
    cpml = cpml / scale

    x_max = DOMAIN_M / 1000.0
    z_max = DOMAIN_M / 1000.0
    extent = [0.0, x_max, z_max, 0.0]

    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.25), facecolor="white")
    cases = [
        (pml, "a.PML边界条件"),
        (cpml, "b.CPML边界条件"),
    ]

    for ax, (field, label) in zip(axes, cases):
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
            -0.18,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            fontproperties=CJK_FONT,
        )
        cbar = fig.colorbar(im, cax=add_right_cax(ax))
        cbar.set_label("Amplitude", fontsize=11)
        cbar.ax.tick_params(labelsize=10)

    fig.subplots_adjust(wspace=0.30, bottom=0.16)
    png = OUT_DIR / "uniform_pml_cpml_compare_0p95s.png"
    pdf = OUT_DIR / "uniform_pml_cpml_compare_0p95s.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    plot_compare()
