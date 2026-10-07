import numpy as np

def compute_gradient_weighted_risk(delta_wepl: np.ndarray, 
                                   mc_dose: np.ndarray, 
                                   beam_axis: int, 
                                   spacing: float) -> np.ndarray:
    """
    Computes the gradient-weighted risk map for proton therapy.
    
    Formula: Risk(x,y,z) = |ΔWEPL(x,y,z)| × |∂D_ref/∂z|
    
    Args:
        delta_wepl: 3D array of accumulated WEPL differences (mm), shape (Z, Y, X).
        mc_dose: 3D array of reference Monte Carlo dose (Gy), shape (Z, Y, X).
        beam_axis: The axis (0=Z, 1=Y, 2=X) along which the beam travels.
        spacing: The voxel spacing (mm) along the beam_axis.
        
    Returns:
        risk_map: 3D array of the same shape containing the risk scores (Gy * mm).
    """
    # Compute the dose gradient along the beam direction.
    # By specifying `axis` as an integer, np.gradient returns a single ndarray.
    dose_gradient = np.gradient(mc_dose, spacing, axis=beam_axis)
        
    abs_dose_gradient = np.abs(dose_gradient)
    abs_delta_wepl = np.abs(delta_wepl)
    
    # Risk = |ΔWEPL| × |∂D/∂z|
    risk_map = abs_delta_wepl * abs_dose_gradient
    
    return risk_map 