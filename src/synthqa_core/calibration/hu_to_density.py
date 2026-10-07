"""
This implements the exact DoseRAD2026 curve using scipy.interpolate.interp1d.
"""
import numpy as np
from scipy.interpolate import interp1d
from typing import Union, Optional

# Exact curve from DoseRAD2026 Appendix B
DOSERAD_HU_DENSITY_CURVE = [
    {"hu": -1024, "density_g_cm3": 1.200000e-03},
    {"hu": -999,  "density_g_cm3": 1.210000e-03},
    {"hu": -200,  "density_g_cm3": 8.043754e-01},
    {"hu": -199,  "density_g_cm3": 8.183035e-01},
    {"hu": -10,   "density_g_cm3": 1.006579e+00},
    {"hu": -9,    "density_g_cm3": 9.966749e-01},
    {"hu": 120,   "density_g_cm3": 1.126553e+00},
    {"hu": 121,   "density_g_cm3": 1.095097e+00},
    {"hu": 3000,  "density_g_cm3": 3.027294e+00},
    {"hu": 4000,  "density_g_cm3": 3.698428e+00}
]

class HUToDensityConverter:
    def __init__(self, curve_data: Optional[list] = None):
        if curve_data is None:
            curve_data = DOSERAD_HU_DENSITY_CURVE
            
        # Sort by HU to ensure monotonic interpolation
        curve_data = sorted(curve_data, key=lambda x: x['hu'])
        
        self.hu_values = np.array([point['hu'] for point in curve_data])
        self.density_values = np.array([point['density_g_cm3'] for point in curve_data])
        
        # Linear interpolation. bounds_error=False allows us to clip out-of-range values safely.
        self.interpolator = interp1d(
            self.hu_values, 
            self.density_values, 
            kind='linear', 
            bounds_error=False, 
            fill_value=(self.density_values[0], self.density_values[-1])
        )

    def convert(self, hu_array: Union[np.ndarray, float, int]) -> np.ndarray:
        """
        Convert Hounsfield Units to Mass Density (g/cm³).
        Clips values to the valid range [-1024, 4000] before interpolation.
        """
        hu_array = np.asarray(hu_array, dtype=np.float32)
        hu_clipped = np.clip(hu_array, self.hu_values.min(), self.hu_values.max())
        return self.interpolator(hu_clipped)