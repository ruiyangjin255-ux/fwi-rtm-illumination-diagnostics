import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path


out_dir = Path(r"D:\ryjin\paper_figures_source\vti\tti_beta_series_vti_settings")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "tti_beta_psi_schematic.png"

beta = np.deg2rad(30.0)
psi = np.deg2rad(60.0)

# Coordinate convention: z is positive downward, matching seismic x-z sections.
v_vti = np.array([0.0, 0.0, 1.0])
v_tti = np.array([
    np.sin(beta) * np.cos(psi),
    np.sin(beta) * np.sin(psi),
    np.cos(beta),
])
proj = np.array([v_tti[0], v_tti[1], 0.0])

fig = plt.figure(figsize=(13, 6), dpi=300)

# ---- 3D view: beta and psi together ----
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.set_title(r"TTI symmetry-axis direction: $\beta=30^\circ,\ \psi=60^\circ$", fontsize=13, pad=12)

lim = 1.1
ax.quiver(0, 0, 0, 1, 0, 0, color="k", arrow_length_ratio=0.08, linewidth=1.6)
ax.quiver(0, 0, 0, 0, 1, 0, color="k", arrow_length_ratio=0.08, linewidth=1.6)
ax.quiver(0, 0, 0, 0, 0, 1, color="k", arrow_length_ratio=0.08, linewidth=1.6)
ax.text(1.08, 0, 0, "x", fontsize=12)
ax.text(0, 1.08, 0, "y", fontsize=12)
ax.text(0, 0, 1.08, "z down", fontsize=12)

ax.quiver(0, 0, 0, *v_vti, color="#1f77b4", arrow_length_ratio=0.08, linewidth=3)
ax.text(0.03, 0.03, 0.92, "VTI axis", color="#1f77b4", fontsize=11)

ax.quiver(0, 0, 0, *v_tti, color="#d62728", arrow_length_ratio=0.08, linewidth=3)
ax.text(*(v_tti * 1.08), "TTI axis", color="#d62728", fontsize=11)

ax.plot([0, proj[0]], [0, proj[1]], [0, 0], color="#d62728", linestyle="--", linewidth=2)
ax.plot([proj[0], v_tti[0]], [proj[1], v_tti[1]], [0, v_tti[2]], color="#d62728", linestyle=":", linewidth=1.8)

# beta arc in the plane spanned by vertical axis and TTI axis
arc_t = np.linspace(0, beta, 60)
arc = np.column_stack([
    0.28 * np.sin(arc_t) * np.cos(psi),
    0.28 * np.sin(arc_t) * np.sin(psi),
    0.28 * np.cos(arc_t),
])
ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color="#d62728", linewidth=2)
ax.text(*(arc[len(arc)//2] * 1.25), r"$\beta$", color="#d62728", fontsize=14)

# psi arc on horizontal plane
psi_t = np.linspace(0, psi, 60)
ax.plot(0.38 * np.cos(psi_t), 0.38 * np.sin(psi_t), np.zeros_like(psi_t),
        color="#2ca02c", linewidth=2)
ax.text(0.30, 0.18, 0.02, r"$\psi$", color="#2ca02c", fontsize=14)

ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_zlim(0, lim)
ax.invert_zaxis()
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.view_init(elev=22, azim=-58)
ax.grid(True, alpha=0.35)

# ---- 2D x-z comparison: VTI vs apparent TTI tilt in a vertical section ----
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title("2D vertical-section interpretation", fontsize=13, pad=12)
ax2.set_aspect("equal")
ax2.set_xlim(-1.05, 1.05)
ax2.set_ylim(1.08, -0.08)

ax2.arrow(0, 0, 0, 0.95, width=0.008, head_width=0.055, head_length=0.07,
          length_includes_head=True, color="#1f77b4")
ax2.text(0.05, 0.52, r"VTI: $\beta=0^\circ$", color="#1f77b4", fontsize=12)

# x-z projection of TTI axis. The apparent section tilt is atan(nx/nz).
apparent_x = v_tti[0]
apparent_z = v_tti[2]
scale = 0.95 / apparent_z
ax2.arrow(0, 0, apparent_x * scale, apparent_z * scale, width=0.008,
          head_width=0.055, head_length=0.07, length_includes_head=True,
          color="#d62728")
ax2.text(apparent_x * scale + 0.02, apparent_z * scale * 0.65,
         r"TTI axis", color="#d62728", fontsize=12)

theta = np.linspace(0, np.arctan2(apparent_x, apparent_z), 60)
ax2.plot(0.25 * np.sin(theta), 0.25 * np.cos(theta), color="#d62728", linewidth=2)
ax2.text(0.10, 0.22, r"apparent tilt in x-z", color="#d62728", fontsize=11)

ax2.axhline(0, color="0.2", linewidth=1)
ax2.axvline(0, color="0.2", linewidth=1)
ax2.text(0.92, -0.02, "x", fontsize=12)
ax2.text(0.03, 1.02, "z down", fontsize=12)
ax2.set_xticks([])
ax2.set_yticks([])
for spine in ax2.spines.values():
    spine.set_visible(False)

fig.tight_layout()
fig.savefig(out_path, bbox_inches="tight")
print(out_path)
