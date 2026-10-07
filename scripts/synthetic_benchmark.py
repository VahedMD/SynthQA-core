"""
Synthetic Benchmark for Proton QA Surrogate Validation

Validates the surrogate using an analytical approach:
1. Create a synthetic CT (water + bone inhomogeneity)
2. Compute reference dose analytically (Bragg peak)
3. Perturb the CT (simulate sCT error)
4. Compute ΔWEPL and gradient-weighted risk
5. Analytically compute the "true" dose error by shifting the Bragg peak
6. Compare predicted risk vs actual dose error (Pearson correlation)
7. Benchmark computation time
"""
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from synthqa_core.calibration.hu_to_density import HUToDensityConverter
from synthqa_core.calibration.density_to_spr import DensityToSPRConverter
from synthqa_core.surrogates.wepl import compute_wepl_along_axis
from synthqa_core.risk.gradient_weighted import compute_gradient_weighted_risk
from synthqa_core.risk.thresholds import classify_risk


def create_synthetic_ct(shape=(100, 100, 100), spacing=(2.0, 2.0, 2.0)):
    """Create a synthetic CT with water background and bone inhomogeneity."""
    ct = np.zeros(shape, dtype=np.float32)  # Water = 0 HU
    
    # Add a bone slab (HU = 1000) in the middle of the Z-axis
    z_start, z_end = 40, 50
    ct[z_start:z_end, 30:70, 30:70] = 1000.0  # Bone
    
    return ct, spacing


def create_synthetic_sct(ct_ref, error_type="regional_bias", error_magnitude=200.0):
    """
    Simulate a synthetic CT with errors.
    
    Args:
        ct_ref: Reference CT (HU values)
        error_type: 'uniform_shift', 'gaussian_noise', or 'regional_bias'
        error_magnitude: Magnitude of the error (HU)
    """
    sct = ct_ref.copy()
    
    if error_type == "uniform_shift":
        sct += error_magnitude
    elif error_type == "gaussian_noise":
        noise = np.random.normal(0, error_magnitude, size=ct_ref.shape)
        sct += noise
    elif error_type == "regional_bias":
        # Underestimate bone density (common sCT failure mode)
        bone_mask = ct_ref > 500
        sct[bone_mask] -= error_magnitude
        
    return sct


def create_analytical_dose(spr_array, spacing_xyz, beam_axis=0, 
                            nominal_range=150.0, max_dose=10.0):
    """
    Create an analytical dose distribution based on SPR map.
    
    Simplified Bragg peak model: dose peaks where cumulative WEPL = nominal_range.
    """
    # Compute WEPL along beam axis
    wepl = compute_wepl_along_axis(spr_array, spacing_xyz, axis=beam_axis)
    
    # Bragg peak: Gaussian centered at nominal_range
    sigma = 5.0  # Width of Bragg peak in mm
    dose = max_dose * np.exp(-0.5 * ((wepl - nominal_range) / sigma) ** 2)
    
    # Entrance dose (lower than peak, ramps up)
    entrance_dose = 0.3 * max_dose
    entrance_mask = wepl < nominal_range
    dose[entrance_mask] = np.maximum(
        dose[entrance_mask], 
        entrance_dose * (wepl[entrance_mask] / nominal_range)
    )
    
    return dose


def run_benchmark():
    """Run the full synthetic benchmark."""
    print("=" * 70)
    print("SYNTHETIC BENCHMARK: Proton QA Surrogate Validation")
    print("=" * 70)
    
    # Step 1: Create synthetic CT
    print("\n[1/7] Creating synthetic CT...")
    ct_ref, spacing = create_synthetic_ct(shape=(100, 100, 100), spacing=(2.0, 2.0, 2.0))
    print(f"  CT shape: {ct_ref.shape}, spacing: {spacing}")
    
    # Step 2: Convert CT → SPR
    print("\n[2/7] Converting CT → Density → SPR...")
    hu_conv = HUToDensityConverter()
    spr_conv = DensityToSPRConverter(method="linear_approx")
    
    density_ref = hu_conv.convert(ct_ref)
    spr_ref = spr_conv.convert(density_ref)
    print(f"  SPR range: [{spr_ref.min():.3f}, {spr_ref.max():.3f}]")
    
    # Step 3: Create reference MC dose (analytical)
    print("\n[3/7] Creating analytical reference dose (MC surrogate)...")
    dose_ref = create_analytical_dose(spr_ref, spacing, beam_axis=0, 
                                       nominal_range=150.0, max_dose=10.0)
    print(f"  Max dose: {dose_ref.max():.2f} Gy")
    
    # Step 4: Simulate sCT error
    print("\n[4/7] Simulating synthetic CT error (regional bone bias)...")
    ct_sct = create_synthetic_sct(ct_ref, error_type="regional_bias", error_magnitude=200.0)
    density_sct = hu_conv.convert(ct_sct)
    spr_sct = spr_conv.convert(density_sct)
    print(f"  sCT error: -200 HU in bone region (density underestimation)")
    
    # Step 5: Compute ΔSPR and ΔWEPL
    print("\n[5/7] Computing ΔSPR and ΔWEPL...")
    delta_spr = spr_sct - spr_ref
    delta_wepl = compute_wepl_along_axis(delta_spr, spacing, axis=0)
    print(f"  Max |ΔWEPL|: {np.abs(delta_wepl).max():.2f} mm")
    
    # Step 6: Compute gradient-weighted risk map
    print("\n[6/7] Computing gradient-weighted risk map...")
    start_time = time.perf_counter()
    
    dose_gradient = np.abs(np.gradient(dose_ref, spacing[2], axis=0))
    risk_map = compute_gradient_weighted_risk(
        delta_wepl=delta_wepl,
        mc_dose=dose_ref,
        beam_axis=0,
        spacing=spacing[2]
    )
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    print(f"  Risk computation time: {elapsed_ms:.2f} ms")
    
    # Step 7: Compute "ground truth" dose error analytically
    print("\n[7/7] Computing analytical 'ground truth' dose error...")
    dose_sct = create_analytical_dose(spr_sct, spacing, beam_axis=0, 
                                       nominal_range=150.0, max_dose=10.0)
    dose_error = np.abs(dose_sct - dose_ref)
    print(f"  Max dose error: {dose_error.max():.2f} Gy")
    
    # Validation
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    
    # Pearson correlation in high-dose regions
    risk_flat = risk_map.flatten()
    error_flat = dose_error.flatten()
    dose_mask = dose_ref.flatten() > 0.05 * dose_ref.max()
    risk_filtered = risk_flat[dose_mask]
    error_filtered = error_flat[dose_mask]
    
    if len(risk_filtered) > 2 and np.std(risk_filtered) > 0:
        correlation = np.corrcoef(risk_filtered, error_filtered)[0, 1]
        print(f"\n  Pearson correlation (risk vs dose error): {correlation:.4f}")
        print(f"  Voxels analyzed: {len(risk_filtered):,}")
    else:
        print("\n  Warning: Not enough variance for correlation analysis")
        correlation = 0.0
    
    # Risk classification
    risk_classes = classify_risk(delta_wepl, dose_gradient, gradient_threshold=0.1)
    high_risk = np.sum(risk_classes == 2)
    med_risk = np.sum(risk_classes == 1)
    
    print(f"\n  High risk voxels (|ΔWEPL| ≥ 3mm): {high_risk:,}")
    print(f"  Medium risk voxels (1-3mm): {med_risk:,}")
    
    # Timing
    print(f"\n  Total surrogate computation: {elapsed_ms:.2f} ms")
    print(f"  Target: < 10 ms")
    print(f"  Status: {'✓ PASSED' if elapsed_ms < 10 else '✗ FAILED'}")
    
    return {
        "correlation": correlation,
        "elapsed_ms": elapsed_ms,
        "high_risk_voxels": high_risk,
        "med_risk_voxels": med_risk
    }


if __name__ == "__main__":
    results = run_benchmark()
    print("\n" + "=" * 70)
    print("Benchmark complete!")
    print("=" * 70)