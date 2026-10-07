import numpy as np
import SimpleITK as sitk
import os
import sys

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from synthqa_core.io.image_loader import load_mha
from synthqa_core.calibration.hu_to_density import HUToDensityConverter

def test_image_loader_axes():
    """Verify that SimpleITK arrays are loaded with the correct Z,Y,X shape and X,Y,Z spacing."""
    # Create a dummy .mha file: 30 voxels X, 20 voxels Y, 10 voxels Z
    dummy_array = np.zeros((10, 20, 30), dtype=np.int16) 
    img = sitk.GetImageFromArray(dummy_array)
    img.SetSpacing((1.0, 1.0, 3.0)) # DoseRAD2026 grid is 1x1x3 mm
    dummy_path = "tests/dummy_ct.mha"
    sitk.WriteImage(img, dummy_path)
    
    # Load via our module
    data = load_mha(dummy_path)
    
    # Assertions
    assert data["shape_zyx"] == (10, 20, 30), "SimpleITK array shape should be (Z, Y, X)"
    assert data["spacing_xyz"] == (1.0, 1.0, 3.0), "Spacing should be (X, Y, Z)"
    
    # Cleanup
    os.remove(dummy_path)

def test_hu_to_density_water_equivalence():
    """Verify the DoseRAD2026 curve handles water correctly."""
    converter = HUToDensityConverter()
    
    # Calculate expected density for HU=0 by interpolating between -9 and 120
    # Expected math: ~1.0057 g/cm³
    density_0 = converter.convert(0)
    
    print(f"\n[TEST] HU=0 maps to {density_0:.4f} g/cm³")
    assert 0.99 < density_0 < 1.02, f"HU=0 density {density_0} is not ≈ 1.0"
    
    # Verify bounds clipping
    assert converter.convert(-2000) == converter.convert(-1024), "Lower bound clipping failed"
    assert converter.convert(5000) == converter.convert(4000), "Upper bound clipping failed"