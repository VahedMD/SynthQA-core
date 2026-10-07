import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from synthqa_core.risk.gradient_weighted import compute_gradient_weighted_risk
from synthqa_core.risk.range_shift import find_distal_edge, predict_range_shift
from synthqa_core.risk.thresholds import classify_risk

def test_gradient_weighted_risk():
    """Test the core risk formula: Risk = |ΔWEPL| × |∂D/∂z|"""
    # 1D setup mapped to 3D (Z, Y, X) to simulate a Z-axis beam
    # Z-axis beam, spacing = 2.0 mm
    # Dose profile: plateau then sharp drop
    dose_1d = np.array([0.0, 2.0, 2.0, 0.0])
    # ΔWEPL: error increases distally (e.g. 2.0 mm error at the edge)
    wepl_err_1d = np.array([0.0, 0.0, 1.0, 2.0])
    
    mc_dose = dose_1d.reshape(4, 1, 1)
    delta_wepl = wepl_err_1d.reshape(4, 1, 1)
    
    risk = compute_gradient_weighted_risk(delta_wepl, mc_dose, beam_axis=0, spacing=2.0)
    
    print(f"\n[TEST] Risk Map (Z-axis): {risk.flatten()}")
    
    # At the distal edge (Z=3), gradient is high, and WEPL error is high -> Risk > 0
    assert risk[-1, 0, 0] > 0, "Distal edge risk should be > 0"
    
    # At entrance (Z=0), WEPL error is 0, so risk must be 0 regardless of gradient
    assert risk[0, 0, 0] == 0, "Entrance region risk should be 0"
    print("[TEST] Gradient-weighted risk calculation passed.")

def test_range_shift():
    """Test distal edge detection (R80)"""
    # Create a pristine Bragg peak
    wepl_axis = np.linspace(0, 150, 1000) # 0 to 150 mm WEPL
    dose = np.exp(-0.01 * (wepl_axis - 100)**2) * 10.0 
    # Add a sharp falloff to mimic distal edge
    dose[wepl_axis > 105] *= np.exp(-2.0 * (wepl_axis[wepl_axis > 105] - 105))
    
    r80_ref, max_dose = find_distal_edge(dose, wepl_axis, threshold_percent=80.0)
    print(f"\n[TEST] Pristine R80 detected at WEPL: {r80_ref:.2f} mm")
    
    # Shift the WEPL axis by 3mm (simulate sCT error)
    wepl_sct = wepl_axis + 3.0
    r80_sct, _ = find_distal_edge(dose, wepl_sct, threshold_percent=80.0)
    print(f"[TEST] Shifted R80 detected at WEPL: {r80_sct:.2f} mm")
    
    shift = predict_range_shift(r80_ref, r80_sct)
    print(f"[TEST] Predicted range shift: {shift:.2f} mm (Expected ~3.0 mm)")
    assert np.isclose(shift, 3.0, atol=0.1), "Range shift prediction failed"

def test_risk_thresholds():
    """Test risk classification based on WEPL and gradient thresholds"""
    # Voxel 0: 0.5mm error, low gradient -> 0 (Safe)
    # Voxel 1: 1.5mm error, high gradient -> 1 (Medium Risk)
    # Voxel 2: 3.5mm error, high gradient -> 2 (High Risk)
    # Voxel 3: 4.0mm error, low gradient -> 0 (Safe! E.g., uniform dose region)
    delta_wepl = np.array([0.5, 1.5, 3.5, 4.0]).reshape(4, 1, 1)
    dose_gradient = np.array([0.0, 0.5, 0.5, 0.0]).reshape(4, 1, 1)
    
    risk_class = classify_risk(delta_wepl, dose_gradient, gradient_threshold=0.1)
    
    print(f"\n[TEST] Risk Classes: {risk_class.flatten()}")
    expected = np.array([0, 1, 2, 0]).reshape(4, 1, 1)
    assert np.array_equal(risk_class, expected), "Risk threshold classification failed"
    print("[TEST] Risk threshold classification passed.")

if __name__ == "__main__":
    test_gradient_weighted_risk()
    test_range_shift()
    test_risk_thresholds()
    print("\nPhase 4 tests passed successfully!")