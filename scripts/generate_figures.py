"""
Generates publication-ready figures for Article 1.

Figure 1: Phantom geometry (anatomy, beam direction, profile columns)
Figure 2: Bragg peak depth-dose profiles (reference CT vs synthetic CT)
Figure 3: Gradient-weighted risk map overlay (mm axes + colorbar)
Figure 4: Scatter plot (Predicted Risk vs Actual Dose Error)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from synthqa_core.calibration.hu_to_density import HUToDensityConverter
from synthqa_core.calibration.density_to_spr import DensityToSPRConverter
from synthqa_core.surrogates.wepl import compute_wepl_along_axis
from synthqa_core.risk.gradient_weighted import compute_gradient_weighted_risk
from synthqa_core.risk.range_shift import find_distal_edge

from synthetic_benchmark import (
    create_synthetic_ct, create_synthetic_sct, create_analytical_dose
)

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "figure.dpi": 150,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')


# ---------------------------------------------------------------- data ----
def build_data():
    ct_ref, spacing = create_synthetic_ct(shape=(100, 100, 100),
                                          spacing=(2.0, 2.0, 2.0))
    hu_conv = HUToDensityConverter()
    spr_conv = DensityToSPRConverter(method="linear_approx")

    spr_ref = spr_conv.convert(hu_conv.convert(ct_ref))
    dose_ref = create_analytical_dose(spr_ref, spacing, beam_axis=0,
                                      nominal_range=150.0, max_dose=10.0)

    ct_sct = create_synthetic_sct(ct_ref, error_type="regional_bias",
                                  error_magnitude=200.0)
    spr_sct = spr_conv.convert(hu_conv.convert(ct_sct))
    dose_sct = create_analytical_dose(spr_sct, spacing, beam_axis=0,
                                      nominal_range=150.0, max_dose=10.0)

    delta_spr = spr_sct - spr_ref
    delta_wepl = compute_wepl_along_axis(delta_spr, spacing, axis=0)
    risk_map = compute_gradient_weighted_risk(delta_wepl, dose_ref,
                                              beam_axis=0, spacing=spacing[2])
    dose_error = np.abs(dose_sct - dose_ref)

    return dict(ct_ref=ct_ref, spacing=spacing, dose_ref=dose_ref,
                dose_sct=dose_sct, risk_map=risk_map, dose_error=dose_error,
                delta_wepl=delta_wepl)


# ------------------------------------------------------- Figure 1 ---------
def fig1_phantom_geometry(d):
    """Phantom anatomy with beam direction and the two profile columns."""
    ct = d["ct_ref"]
    sx, sy, sz = d["spacing"]
    nz, ny, nx = ct.shape
    mid_y = ny // 2
    ct_slice = ct[:, mid_y, :]
    extent = [0, nx * sx, nz * sz, 0]          # FIX #1: true mm axes

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(ct_slice, cmap='bone', vmin=-1024, vmax=2000, extent=extent)
    fig.colorbar(im, ax=ax, label="Hounsfield Units (HU)")

    # Bone slab outline (indices 40:50 z, 30:70 x  ->  80-100 mm z, 60-140 mm x)
    ax.add_patch(Rectangle((60, 80), 80, 20, linewidth=2,
                           edgecolor='yellow', facecolor='none'))
    ax.text(100, 90, "Bone slab\n(HU = 1000)", ha='center', va='center',
            color='yellow', fontsize=11, weight='bold')
    ax.text(160, 40, "Water\n(HU = 0)", ha='center', color='white', fontsize=11)

    # Beam direction arrow
    ax.annotate("", xy=(10, 190), xytext=(10, 15),
                arrowprops=dict(color='lime', width=2, headwidth=10))
    ax.text(14, 100, "Beam\ndirection", color='lime', fontsize=11, weight='bold')

    # Profile columns used in Figure 2
    ax.axvline(100, color='red', ls='--', lw=1.5)
    ax.axvline(20, color='cyan', ls='--', lw=1.5)
    ax.legend(handles=[
        Line2D([0], [0], color='red', ls='--', label='Profile A (through bone)'),
        Line2D([0], [0], color='cyan', ls='--', label='Profile B (water only)'),
    ], loc='upper right')

    ax.set_title("Figure 1: Phantom Geometry (Coronal Plane)")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z — beam direction (mm)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Fig1_Phantom_Geometry.png"),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved Fig1_Phantom_Geometry.png")


# ------------------------------------------------------- Figure 2 ---------
def fig2_bragg_profiles(d):
    """Depth-dose (Bragg peak) profiles: reference vs synthetic CT."""
    dose_ref, dose_sct = d["dose_ref"], d["dose_sct"]
    sx, sy, sz = d["spacing"]
    nz, ny, nx = dose_ref.shape
    mid_y = ny // 2
    z_mm = (np.arange(nz) + 0.5) * sz

    xA, xB = 50, 10   # voxel columns: through bone (100 mm) / water only (20 mm)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    for ax, xc, title in [(axes[0], xA, "(a) Profile A — ray through bone slab"),
                          (axes[1], xB, "(b) Profile B — ray through water only")]:
        ref = dose_ref[:, mid_y, xc]
        sct = dose_sct[:, mid_y, xc]

        ax.plot(z_mm, ref, 'b-', lw=2, label='Reference CT')
        ax.plot(z_mm, sct, 'r--', lw=2, label='Synthetic CT')

        r80_ref, _ = find_distal_edge(ref, z_mm)
        r80_sct, _ = find_distal_edge(sct, z_mm)
        shift = r80_sct - r80_ref

        ax.axvline(r80_ref, color='b', ls=':', alpha=0.5)
        ax.axvline(r80_sct, color='r', ls=':', alpha=0.5)
        ax.annotate("", xy=(r80_sct, 5.0), xytext=(r80_ref, 5.0),
                    arrowprops=dict(arrowstyle="<->", color="k", lw=1.5))
        ax.text((r80_ref + r80_sct) / 2, 5.8,
                f"$\\Delta R_{{80}}$ = {shift:.2f} mm", ha='center', fontsize=11)

        ax.set_title(title)
        ax.set_xlabel("Depth Z (mm)")
        ax.grid(True, ls='--', alpha=0.3)

    axes[0].set_ylabel("Dose (Gy)")
    axes[0].legend(loc='upper left')
    fig.suptitle("Figure 2: Bragg Peak Profiles — sCT Error Shifts the Distal Edge "
                 "Only Behind the Heterogeneity", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Fig2_Bragg_Profiles.png"),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved Fig2_Bragg_Profiles.png")


# ------------------------------------------------------- Figure 3 ---------
def fig3_risk_overlay(d):
    """Risk map overlay with mm axes, colorbar, and unlabeled R80 contours."""
    ct, risk = d["ct_ref"], d["risk_map"]
    dose_ref, dose_sct = d["dose_ref"], d["dose_sct"]
    sx, sy, sz = d["spacing"]
    nz, ny, nx = ct.shape
    mid_y = ny // 2
    extent = [0, nx * sx, nz * sz, 0]          # FIX #1: true mm axes

    ct_slice = ct[:, mid_y, :]
    risk_slice = risk[:, mid_y, :]
    dose_slice_ref = dose_ref[:, mid_y, :]
    dose_slice_sct = dose_sct[:, mid_y, :]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(ct_slice, cmap='bone', vmin=-1024, vmax=2000, extent=extent)

    cmap_risk = LinearSegmentedColormap.from_list(
        'risk', ['black', 'red', 'yellow', 'white'])
    risk_masked = np.ma.masked_where(risk_slice < 0.01, risk_slice)
    im2 = ax.imshow(risk_masked, cmap=cmap_risk, alpha=0.6, extent=extent)
    fig.colorbar(im2, ax=ax, label="Risk  |$Delta$WEPL| $\\times$ |$\\nabla$D|  (Gy$\\cdot$mm)")

    # FIX #2: contours WITHOUT clabel; identified via legend proxies instead
    xx, zz = np.meshgrid((np.arange(nx) + 0.5) * sx, (np.arange(nz) + 0.5) * sz)
    ax.contour(xx, zz, dose_slice_ref, levels=[0.8 * dose_slice_ref.max()],
               colors='cyan', linewidths=1.5)
    ax.contour(xx, zz, dose_slice_sct, levels=[0.8 * dose_slice_sct.max()],
               colors='magenta', linewidths=1.5)

    ax.legend(handles=[
        Line2D([0], [0], color='cyan', lw=1.5, label='R80 — reference CT'),
        Line2D([0], [0], color='magenta', lw=1.5, label='R80 — synthetic CT'),
    ], loc='upper right')

    ax.set_title("Figure 3: Gradient-Weighted Risk Map (Coronal Slice)")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z — beam direction (mm)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Fig3_Risk_Overlay.png"),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved Fig3_Risk_Overlay.png")


# ------------------------------------------------------- Figure 4 ---------
def fig4_scatter(d):
    """Predicted risk vs actual dose error, colored by depth."""
    risk, err, dose_ref = d["risk_map"], d["dose_error"], d["dose_ref"]
    sz = d["spacing"][2]

    z_coords = np.indices(risk.shape)[0].flatten() * sz
    mask = dose_ref.flatten() > 0.05 * dose_ref.max()
    r = risk.flatten()[mask]
    e = err.flatten()[mask]
    z = z_coords[mask]

    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(r, e, c=z, cmap='plasma', alpha=0.6, edgecolors='none', s=15)

    coef = np.polyfit(r, e, 1)
    p = np.poly1d(coef)
    r2 = 1 - np.sum((e - p(r)) ** 2) / np.sum((e - np.mean(e)) ** 2)
    ax.plot(np.sort(r), p(np.sort(r)), "r--", lw=2, alpha=0.8,
            label=f"Linear fit (R$^2$ = {r2:.3f})")

    ax.set_xlabel("Predicted Risk (Gy$\\cdot$mm)")
    ax.set_ylabel("Actual Dose Error (Gy)")
    ax.set_title("Figure 4: Surrogate Predictive Power vs Ground Truth")
    ax.grid(True, ls='--', alpha=0.3)
    ax.legend(fontsize=12)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Depth Z (mm)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "Fig4_Scatter_Correlation.png"),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Saved Fig4_Scatter_Correlation.png")


# ------------------------------------------------------------- main -------
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Building data...")
    data = build_data()

    print("Generating figures...")
    fig1_phantom_geometry(data)
    fig2_bragg_profiles(data)
    fig3_risk_overlay(data)
    fig4_scatter(data)

    print("\nSuggested manuscript captions:")
    print(" Fig 1: Synthetic phantom (coronal plane). The proton beam enters from the top;")
    print("        dashed lines mark the two analysis rays (A: through bone, B: water only).")
    print(" Fig 2: Depth-dose profiles. The -200 HU bone bias in the sCT shifts the distal")
    print("        edge distally by ~2.7 mm along ray A, while ray B is unaffected.")
    print(" Fig 3: Gradient-weighted risk map. Cyan/magenta contours: R80 of reference/synthetic")
    print("        dose. Risk concentrates on the shifted distal edge behind the slab.")
    print(" Fig 4: Voxel-wise correlation between predicted risk and actual dose error (R^2).")
    print("\nFigure generation complete!")