"""
Spatial Agreement V4: Ray-by-ray dose shift evaluated at the Bragg peak.
Validates the 'localizes' claim using Dice, sensitivity, specificity.
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from synthqa_core.calibration.profiles import CalibrationProfile

def dice(a, b):
    s = np.sum(a) + np.sum(b)
    return 2 * np.sum(a & b) / s if s > 0 else 0.0

def compute_spatial_metrics(risk_map, dose_error, thresh_pct=5):
    """Binarize at the top (100-thresh_pct)% of each map."""
    r_thresh = np.percentile(risk_map, 100 - thresh_pct)
    e_thresh = np.percentile(dose_error, 100 - thresh_pct)
    
    # Fallback for zero maps
    if r_thresh == 0: r_thresh = risk_map.max() * 0.05 if risk_map.max() > 0 else 1.0
    if e_thresh == 0: e_thresh = dose_error.max() * 0.05 if dose_error.max() > 0 else 1.0
    
    risk_bin = risk_map > r_thresh
    err_bin = dose_error > e_thresh
    
    d = dice(risk_bin, err_bin)
    sens = np.sum(risk_bin & err_bin) / np.sum(err_bin) if np.sum(err_bin) > 0 else 0.0
    spec = np.sum(~risk_bin & ~err_bin) / np.sum(~err_bin) if np.sum(~err_bin) > 0 else 0.0
    
    return {"dice": d, "sens": sens, "spec": spec}

def ray_by_ray_shift(dose_ref, delta_wepl, axis=0):
    """
    Vectorized ray-by-ray shift.
    Samples delta_wepl exactly at the Bragg peak index of each ray.
    """
    d = np.moveaxis(dose_ref, axis, 0)
    w = np.moveaxis(delta_wepl, axis, 0)
    
    n_z = d.shape[0]
    z_grid = np.arange(n_z)
    
    # Find the peak index for each ray
    peak_idx = np.argmax(d, axis=0)
    
    # Get the WEPL error exactly at the Bragg peak for each ray
    shifts = np.take_along_axis(w, peak_idx[None, ...], axis=0)[0]
    
    # Only shift rays that actually have a Bragg peak
    has_peak = d.max(axis=0) > 0
    shifts = shifts * has_peak
    
    d_sct = np.zeros_like(d)
    for i in range(d.shape[1]):
        for j in range(d.shape[2]):
            if has_peak[i, j]:
                # np.interp evaluates at (z + shift). 
                # If shift is negative (bone underestimation), it samples at z - |shift|, 
                # shifting the peak distally (to a larger z index).
                d_sct[:, i, j] = np.interp(z_grid + shifts[i, j], z_grid, d[:, i, j], left=0.0, right=0.0)
                
    return np.moveaxis(d_sct, 0, axis)

def test_phantom():
    print("\n" + "="*70)
    print("[SPATIAL AGREEMENT V4 - CORRECT PEAK SAMPLING]")
    print("="*70)
    print("SPATIAL AGREEMENT: Synthetic Phantom (Z-axis beam)")
    print("="*70)
    
    shape = (100, 100, 100)
    spacing = (2.0, 2.0, 2.0)
    ct = np.zeros(shape, dtype=np.float32)
    ct[40:50, 30:70, 40:60] = 1000.0  # bone slab (Z=40..50, Y=30..70, X=40..60)
    
    prof = CalibrationProfile()
    spr_ref = prof.density_to_spr.convert(prof.hu_to_density.convert(ct))
    
    sct = ct.copy()
    sct[sct > 300] -= 200 # Bone underestimation -> SPR decreases
    spr_sct = prof.density_to_spr.convert(prof.hu_to_density.convert(sct))
    
    # Analytical dose (Bragg peak at Z=75, physical depth 150mm)
    z = np.arange(shape[0]) * spacing[0]
    dose_ref = np.exp(-0.5 * ((z - 150) / 5.0)**2)[:, None, None] * np.ones(shape)
    
    delta_spr = spr_sct - spr_ref
    # WEPL is cumulative sum along Z (axis 0). It will be NEGATIVE after the bone.
    delta_wepl = np.cumsum(delta_spr, axis=0) * spacing[0]
    
    # Risk map = |ΔWEPL| * |∂D/∂z|
    grad = np.abs(np.gradient(dose_ref, spacing[0], axis=0))
    risk = np.abs(delta_wepl) * grad
    
    # RAY-BY-RAY dose shift (samples WEPL error exactly at the Bragg peak)
    dose_sct = ray_by_ray_shift(dose_ref, delta_wepl, axis=0)
    dose_err = np.abs(dose_sct - dose_ref)
    
    print(f"Max dose error: {dose_err.max():.4f} Gy")
    print(f"Max risk value:  {risk.max():.4f}")
    
    # Evaluate on the central coronal slice (Y=50)
    mid = shape[1] // 2
    r_slice = risk[:, mid, :]
    e_slice = dose_err[:, mid, :]
    
    m = compute_spatial_metrics(r_slice, e_slice, thresh_pct=5)
    
    print(f"\nCentral Slice Metrics (Threshold: top 95th percentile):")
    print(f"  Dice:        {m['dice']:.3f}")
    print(f"  Sensitivity: {m['sens']:.3f}")
    print(f"  Specificity: {m['spec']:.3f}")
    
    r_thresh = np.percentile(r_slice, 95)
    e_thresh = np.percentile(e_slice, 95)
    
    r_count = np.sum(r_slice > r_thresh)
    e_count = np.sum(e_slice > e_thresh)
    print(f"  Risk voxels (top 5%):   {r_count}")
    print(f"  Error voxels (top 5%):  {e_count}")

        # ---------------- Visualization with colorbars ----------------
    from matplotlib.colors import ListedColormap, BoundaryNorm

    # mm extents for the (Z, X) coronal slice
    ext = [0, shape[2] * spacing[2], shape[0] * spacing[0], 0]

    r_thresh = np.percentile(r_slice, 95)
    e_thresh = np.percentile(e_slice, 95)
    r_bin = r_slice > r_thresh
    e_bin = e_slice > e_thresh

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.6))

    # Panel 1: predicted risk
    im0 = axes[0].imshow(r_slice, cmap='inferno', aspect='auto', extent=ext)
    axes[0].set_title("(a) Predicted risk map")
    cb0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cb0.set_label("Risk  |ΔWEPL|×|∇D|  (Gy·mm)")

    # Panel 2: actual dose error
    im1 = axes[1].imshow(e_slice, cmap='hot', aspect='auto', extent=ext)
    axes[1].set_title("(b) Actual dose error |ΔD|")
    cb1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cb1.set_label("Dose error (Gy)")

    # Panel 3: 4-class agreement map
    agree = np.zeros(r_slice.shape, dtype=int)
    agree[r_bin] = 1                 # risk only  (false positive)
    agree[e_bin] = 2                 # error only (false negative)
    agree[r_bin & e_bin] = 3         # both       (true positive)
    cmap3 = ListedColormap(["#23272e", "#e8a33d", "#d95f5f", "#4caf7d"])
    norm3 = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap3.N)
    im2 = axes[2].imshow(agree, cmap=cmap3, norm=norm3, aspect='auto', extent=ext)
    axes[2].set_title(f"(c) Spatial agreement (Dice = {m['dice']:.3f})")
    cb2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
    cb2.set_ticklabels(["Neither", "Risk only (FP)", "Error only (FN)", "Both (TP)"])

    for ax in axes:
        ax.set_xlabel("X (mm)")
    axes[0].set_ylabel("Z — beam direction (mm)")

    fig.suptitle("Spatial localization of sCT-induced dose error (bone −200 HU)", y=1.02)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/Fig12_Spatial_Overlap.png", dpi=300, bbox_inches='tight')
    print("\nSaved figures/Fig12_Spatial_Overlap.png")
    
    
    return m

def main():
    m_p = test_phantom()
    
    print("\n" + "="*70)
    print("SPATIAL AGREEMENT SUMMARY (V4)")
    print("="*70)
    print(f"{'Metric':<15} {'Phantom':>10}")
    print("-" * 25)
    for k in ["dice", "sens", "spec"]:
        print(f"{k:<15} {m_p[k]:.3f}")

if __name__ == "__main__":
    main()