"""
This module handles the critical axis mapping between SimpleITK's physical 
space (X, Y, Z) and NumPy's array space (Z, Y, X).
"""
import SimpleITK as sitk
import numpy as np
from typing import Dict

def load_mha(file_path: str) -> Dict:
    """
    Load a MetaImage (.mha) file using SimpleITK.
    
    Args:
        file_path (str): Path to the .mha file.
        
    Returns:
        dict: Dictionary containing the numpy array, spacing, origin, direction, and shape.
              WARNING: array is (Z, Y, X) but spacing/origin are (X, Y, Z).
    """
    img = sitk.ReadImage(file_path)
    
    # GetArrayFromImage returns (Z, Y, X)
    array = sitk.GetArrayFromImage(img)  
    
    # GetSpacing/GetOrigin return (X, Y, Z)
    spacing = img.GetSpacing()           
    origin = img.GetOrigin()             
    direction = img.GetDirection()       
    
    return {
        "array": array,
        "spacing_xyz": spacing,
        "origin_xyz": origin,
        "direction": direction,
        "shape_zyx": array.shape
    }