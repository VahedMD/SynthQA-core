import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from synthqa_core.calibration.hu_to_density import HUToDensityConverter
from synthqa_core.calibration.density_to_spr import DensityToSPRConverter
from synthqa_core.surrogates.wepl import compute_wepl_along_axis, compute_wepl_along_ray

def test_spr_conversion():
    hu_conv = HUToDensityConverter()
    spr_conv = DensityToSPRConverter(method="linear_approx")
    
    # Test water (HU ~ 0 -> density ~ 1.0 -> SPR ~ 1.0)
    density = hu_conv.convert(0)
    spr = spr_conv.convert(density)
    print(f"\n[TEST] HU=0 -> Density={density:.4f} -> SPR={spr:.4f}")
    assert 0.99 < spr < 1.02, "Water SPR should be ~1.0"
    
    # Test bone (HU 1000 -> density ~ 1.5+ -> SPR ~ 1.5+)
    density_bone = hu_conv.convert(1000)
    spr_bone = spr_conv.convert(density_bone)
    print(f"[TEST] HU=1000 -> Density={density_bone:.4f} -> SPR={spr_bone:.4f}")
    assert spr_bone > 1.1, "Bone SPR should be > 1.0"

def test_axis_aligned_wepl():
    # Create a 3x3x3 phantom with uniform SPR = 1.0 (water)
    # Spacing: 2mm X, 2mm Y, 2mm Z
    spr_array = np.ones((3, 3, 3), dtype=np.float32)
    spacing_xyz = (2.0, 2.0, 2.0)
    
    # WEPL along Z axis (axis=0). Each voxel is 2mm.
    # Cumulative sum should be [2.0, 4.0, 6.0] along the Z axis.
    wepl_z = compute_wepl_along_axis(spr_array, spacing_xyz, axis=0)
    assert np.allclose(wepl_z[:, 0, 0], [2.0, 4.0, 6.0]), "Axis-aligned WEPL (Z) failed"
    print("\n[TEST] Axis-aligned WEPL (Z) passed.")
    
    # WEPL along X axis (axis=2).
    wepl_x = compute_wepl_along_axis(spr_array, spacing_xyz, axis=2)
    assert np.allclose(wepl_x[0, 0, :], [2.0, 4.0, 6.0]), "Axis-aligned WEPL (X) failed"
    print("[TEST] Axis-aligned WEPL (X) passed.")

def test_arbitrary_ray_wepl():
    # Create a 10x10x10 water phantom (SPR=1.0)
    # Physical size: 10mm x 10mm x 10mm (Spacing 1x1x1 mm, Origin 0,0,0)
    spr_array = np.ones((10, 10, 10), dtype=np.float32)
    spacing_xyz = (1.0, 1.0, 1.0)
    origin_xyz = (0.0, 0.0, 0.0)
    
    # Ray from (0,0,0) to (9,0,0) -> travels purely along X axis, length 9mm
    # Expected WEPL at end = 9.0 mm (since SPR=1.0)
    ray_source = np.array([0.0, 0.0, 0.0])
    ray_target = np.array([9.0, 0.0, 0.0])
    
    wepl_ray = compute_wepl_along_ray(spr_array, spacing_xyz, origin_xyz, ray_source, ray_target, num_samples=100)
    
    # The final cumulative WEPL should be approximately 9.0
    print(f"\n[TEST] Ray final WEPL: {wepl_ray[-1]:.4f} mm (Expected ~9.0)")
    assert np.isclose(wepl_ray[-1], 9.0, atol=0.1), "Arbitrary ray WEPL failed"
    
    # Test diagonal ray from (0,0,0) to (9,9,9) -> length = sqrt(9^2*3) = 15.588 mm
    ray_target_diag = np.array([9.0, 9.0, 9.0])
    wepl_diag = compute_wepl_along_ray(spr_array, spacing_xyz, origin_xyz, ray_source, ray_target_diag, num_samples=200)
    expected_dist = np.linalg.norm(ray_target_diag - ray_source)
    print(f"[TEST] Diagonal Ray final WEPL: {wepl_diag[-1]:.4f} mm (Expected ~{expected_dist:.4f})")
    assert np.isclose(wepl_diag[-1], expected_dist, atol=0.2), "Diagonal arbitrary ray WEPL failed"

if __name__ == "__main__":
    test_spr_conversion()
    test_axis_aligned_wepl()
    test_arbitrary_ray_wepl()
    print("\nPhase 2 & 3 tests passed successfully!")