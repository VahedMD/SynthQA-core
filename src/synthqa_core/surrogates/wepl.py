"""
Implements both axis-aligned WEPL (using fast numpy.cumsum) 
and arbitrary-angle WEPL (using scipy.ndimage.map_coordinates 
for ray tracing).
"""

import numpy as np
from scipy.ndimage import map_coordinates
from typing import Tuple

def compute_wepl_along_axis(spr_array: np.ndarray, spacing_xyz: Tuple[float, float, float], axis: int) -> np.ndarray:
    """
    Computes WEPL along one of the principal axes (0=SI/Z, 1=AP/Y, 2=LR/X).
    Assumes beam enters from index 0 and travels towards max index.
    
    Args:
        spr_array: 3D array, shape (Z, Y, X).
        spacing_xyz: Tuple (sx, sy, sz).
        axis: 0 for Z, 1 for Y, 2 for X.
    """
    spacing_map = {0: spacing_xyz[2], 1: spacing_xyz[1], 2: spacing_xyz[0]}
    voxel_size = spacing_map[axis]
    
    # Cumulative sum along the axis multiplied by physical voxel size
    wepl = np.cumsum(spr_array, axis=axis) * voxel_size
    return wepl

def compute_wepl_along_ray(spr_array: np.ndarray, spacing_xyz: Tuple[float, float, float], 
                           origin_xyz: Tuple[float, float, float],
                           ray_source: np.ndarray, ray_target: np.ndarray, 
                           num_samples: int = 500) -> np.ndarray:
    """
    Computes cumulative WEPL along a specific arbitrary ray path using trilinear interpolation.
    """
    # 1. Generate points along the ray in physical space
    t = np.linspace(0, 1, num_samples)
    ray_points_x = ray_source[0] + t * (ray_target[0] - ray_source[0])
    ray_points_y = ray_source[1] + t * (ray_target[1] - ray_source[1])
    ray_points_z = ray_source[2] + t * (ray_target[2] - ray_source[2])
    
    # 2. Convert physical coordinates to continuous voxel indices
    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz
    
    # Array is (Z, Y, X), so we need indices in that order for map_coordinates
    iz = (ray_points_z - oz) / sz
    iy = (ray_points_y - oy) / sy
    ix = (ray_points_x - ox) / sx
    
    coordinates = np.array([iz, iy, ix])
    
    # 3. Sample SPR using trilinear interpolation (order=1)
    # mode='constant', cval=0.0 assumes SPR of 0 (vacuum) outside the image bounds
    spr_samples = map_coordinates(spr_array, coordinates, order=1, mode='constant', cval=0.0)
    
    # 4. Calculate step size in mm
    dist = np.linalg.norm(ray_target - ray_source)
    step_size = dist / (num_samples - 1)
    
    # 5. Cumulative sum * step_size = WEPL
    wepl = np.cumsum(spr_samples) * step_size
    
    return wepl