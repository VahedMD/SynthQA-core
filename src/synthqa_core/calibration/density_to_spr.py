"""
Converts mass density (g/cm3) to Relative Stopping Power Ratio (SPR). 
I implemented the simplified linear approximation as the default, 
with hooks for future stoichiometric methods.
"""

import numpy as np
from typing import Union

class DensityToSPRConverter:
    """
    Converts Mass Density (g/cm³) to Relative Stopping Power Ratio (SPR) for protons.
    """
    def __init__(self, method: str = "linear_approx"):
        """
        Args:
            method: 'linear_approx' (SPR ≈ density) or 'schneider' (placeholder for stoichiometric).
        """
        self.method = method
        
    def convert(self, density_array: Union[np.ndarray, float, int]) -> np.ndarray:
        density_array = np.asarray(density_array, dtype=np.float32)
        
        if self.method == "linear_approx":
            # Simplified approach: SPR ≈ mass_density × 1.0 (water = 1.0 g/cm³ ≈ SPR 1.0)
            # Note: This is a first-order approximation valid for soft tissues.
            return density_array
        else:
            raise NotImplementedError(f"Method '{self.method}' not yet implemented.")