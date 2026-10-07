"""
Classifies clinical risk, strictly enforcing the rule that high WEPL 
errors only matter if they occur in high-dose-gradient regions (distal edge).
"""
import numpy as np

def classify_risk(delta_wepl: np.ndarray, 
                  dose_gradient: np.ndarray, 
                  gradient_threshold: float = 0.1) -> np.ndarray:
    """
    Classifies voxels into Low, Medium, and High risk based on ΔWEPL thresholds,
    ONLY flagging them if the dose gradient is significant (distal edge).
    
    Thresholds:
    - Low risk:   |ΔWEPL| < 1.0 mm
    - Medium risk: 1.0 <= |ΔWEPL| < 3.0 mm
    - High risk:  |ΔWEPL| >= 3.0 mm
    
    Args:
        delta_wepl: 3D array of WEPL differences (mm).
        dose_gradient: 3D array of absolute dose gradients (Gy/mm).
        gradient_threshold: Minimum gradient (Gy/mm) required to consider a voxel "at risk".
                            
    Returns:
        risk_class: 3D integer array where:
                    0 = Low/No Risk (gradient too low OR |ΔWEPL| < 1mm)
                    1 = Medium Risk
                    2 = High Risk
    """
    abs_delta_wepl = np.abs(delta_wepl)
    is_high_gradient = dose_gradient >= gradient_threshold
    
    risk_class = np.zeros_like(abs_delta_wepl, dtype=np.int8)
    
    # Medium Risk
    medium_mask = (abs_delta_wepl >= 1.0) & (abs_delta_wepl < 3.0) & is_high_gradient
    risk_class[medium_mask] = 1
    
    # High Risk
    high_mask = (abs_delta_wepl >= 3.0) & is_high_gradient
    risk_class[high_mask] = 2
    
    return risk_class