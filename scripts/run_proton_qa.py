"""
Main CLI entry point for Proton QA Surrogate.
Usage: python scripts/run_proton_qa.py --ct path/to/ct.mha --dose path/to/dose.mha --plan path/to/plan.json
"""
import argparse
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from synthqa_core.io.image_loader import load_mha
from synthqa_core.calibration.hu_to_density import HUToDensityConverter
from synthqa_core.calibration.density_to_spr import DensityToSPRConverter
from synthqa_core.surrogates.wepl import compute_wepl_along_axis
from synthqa_core.risk.gradient_weighted import compute_gradient_weighted_risk
from synthqa_core.risk.thresholds import classify_risk


def run_qa(ct_path: str, dose_path: str, sct_path: str = None):
    """Run the full QA pipeline."""
    print("=" * 70)
    print("PROTON QA SURROGATE - Article 1")
    print("=" * 70)
    
    # Load reference CT
    print("\n[1/6] Loading reference CT...")
    ct_data = load_mha(ct_path)
    ct_array = ct_data["array"]
    spacing = ct_data["spacing_xyz"]
    print(f"  Shape: {ct_data['shape_zyx']}, Spacing: {spacing}")
    
    # Convert to SPR
    print("\n[2/6] Converting CT → SPR...")
    hu_conv = HUToDensityConverter()
    spr_conv = DensityToSPRConverter(method="linear_approx")
    density_ref = hu_conv.convert(ct_array)
    spr_ref = spr_conv.convert(density_ref)
    print(f"  SPR range: [{spr_ref.min():.3f}, {spr_ref.max():.3f}]")
    
    # Load MC dose
    print("\n[3/6] Loading MC dose...")
    dose_data = load_mha(dose_path)
    mc_dose = dose_data["array"]
    print(f"  Dose shape: {dose_data['shape_zyx']}")
    
    # Load or simulate sCT
    if sct_path:
        print("\n[4/6] Loading synthetic CT...")
        sct_data = load_mha(sct_path)
        sct_array = sct_data["array"]
    else:
        print("\n[4/6] No sCT provided. Simulating 100 HU uniform shift...")
        sct_array = ct_array + 100.0
    
    density_sct = hu_conv.convert(sct_array)
    spr_sct = spr_conv.convert(density_sct)
    
    # Compute ΔWEPL
    print("\n[5/6] Computing ΔWEPL...")
    delta_spr = spr_sct - spr_ref
    # Assume Z-axis beam for simplicity (axis=0)
    delta_wepl = compute_wepl_along_axis(delta_spr, spacing, axis=0)
    print(f"  Max |ΔWEPL|: {np.abs(delta_wepl).max():.2f} mm")
    
    # Compute risk map
    print("\n[6/6] Computing gradient-weighted risk map...")
    start_time = time.perf_counter()
    
    risk_map = compute_gradient_weighted_risk(
        delta_wepl=delta_wepl,
        mc_dose=mc_dose,
        beam_axis=0,
        spacing=spacing[2]  # Z-axis spacing
    )
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    # Classify risk
    dose_gradient = np.abs(np.gradient(mc_dose, spacing[2], axis=0))
    risk_classes = classify_risk(delta_wepl, dose_gradient, gradient_threshold=0.1)
    
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"  Computation time: {elapsed_ms:.2f} ms")
    print(f"  Target: < 10 ms")
    print(f"  Status: {'✓ PASSED' if elapsed_ms < 10 else '✗ FAILED'}")
    print(f"  High risk voxels: {np.sum(risk_classes == 2):,}")
    print(f"  Medium risk voxels: {np.sum(risk_classes == 1):,}")
    
    return risk_map, risk_classes, elapsed_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Proton QA Surrogate")
    parser.add_argument("--ct", required=True, help="Path to reference CT (.mha)")
    parser.add_argument("--dose", required=True, help="Path to MC dose (.mha)")
    parser.add_argument("--sct", default=None, help="Path to synthetic CT (.mha)")
    args = parser.parse_args()
    
    run_qa(args.ct, args.dose, args.sct)