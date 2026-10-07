"""
Manages calibration profiles, allowing users to swap between the DoseRAD2026 curve, 
Schneider stoichiometric curves, or custom JSON configurations.
"""
import json
import os
from typing import Optional
from .hu_to_density import HUToDensityConverter, DOSERAD_HU_DENSITY_CURVE
from .density_to_spr import DensityToSPRConverter

class CalibrationProfile:
    def __init__(self, profile_name: str = "doserad2026_default"):
        self.name = profile_name
        self.hu_to_density = HUToDensityConverter(DOSERAD_HU_DENSITY_CURVE)
        self.density_to_spr = DensityToSPRConverter(method="linear_approx")
        
    @classmethod
    def from_json(cls, json_path: str) -> "CalibrationProfile":
        """Load a custom calibration profile from a JSON file."""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Profile JSON not found: {json_path}")
            
        with open(json_path, 'r') as f:
            config = json.load(f)
            
        profile = cls(profile_name=config.get("name", "custom"))
        
        if "hu_to_density_curve" in config:
            profile.hu_to_density = HUToDensityConverter(config["hu_to_density_curve"])
            
        if "density_to_spr_method" in config:
            profile.density_to_spr = DensityToSPRConverter(method=config["density_to_spr_method"])
            
        return profile