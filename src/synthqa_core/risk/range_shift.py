"""
Handles the extraction of the distal edge (e.g., R80 or R90) 
using linear interpolation for sub-voxel accuracy.
"""
import numpy as np
from typing import Tuple


def find_distal_edge(dose_profile: np.ndarray, 
                     wepl_profile: np.ndarray, 
                     threshold_percent: float = 80.0) -> Tuple[float, float]:
    """
    Finds the distal edge position (e.g., R80) along a 1D dose profile.
    
    Args:
        dose_profile: 1D array of dose values along a ray.
        wepl_profile: 1D array of cumulative WEPL values along the same ray (in mm).
        threshold_percent: Percentage of max dose to define the edge (default 80%).
        
    Returns:
        Tuple containing (distal_edge_wepl, max_dose_value).
    """
    max_dose = np.max(dose_profile)
    if max_dose <= 0:
        return np.nan, 0.0
        
    threshold_val = max_dose * (threshold_percent / 100.0)
    
    # Find the peak and look distal to it (from peak_idx to end)
    peak_idx = np.argmax(dose_profile)
    distal_dose = dose_profile[peak_idx:]
    
    # Find where dose drops below threshold
    below_threshold = np.where(distal_dose < threshold_val)[0]
    
    if len(below_threshold) > 0:
        edge_idx_absolute = peak_idx + below_threshold[0]
        
        # Linear interpolation for sub-voxel accuracy
        if 0 < edge_idx_absolute < len(dose_profile):
            x1, x2 = wepl_profile[edge_idx_absolute - 1], wepl_profile[edge_idx_absolute]
            y1, y2 = dose_profile[edge_idx_absolute - 1], dose_profile[edge_idx_absolute]
            
            if y1 != y2:
                slope = (y2 - y1) / (x2 - x1)
                exact_wepl = x1 + (threshold_val - y1) / slope
                return exact_wepl, max_dose
                
        # Fallback to exact voxel WEPL
        return wepl_profile[edge_idx_absolute], max_dose
        
    return np.nan, max_dose

def predict_range_shift(wepl_ref: float, wepl_sct: float) -> float:
    """Predicts the range shift (in mm). Positive means sCT has shorter range."""
    return wepl_sct - wepl_ref