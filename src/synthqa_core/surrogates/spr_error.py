"""
Handles the differences between the reference CT and the synthetic CT.
"""

import numpy as np

def compute_spr_error(spr_ref: np.ndarray, spr_sct: np.ndarray) -> np.ndarray:
    """
    Computes the voxel-wise SPR difference between reference CT and synthetic CT.
    ΔSPR = SPR_sCT - SPR_ref
    """
    return spr_sct - spr_ref

def compute_delta_wepl_along_axis(delta_spr: np.ndarray, spacing_xyz: tuple, axis: int) -> np.ndarray:
    """
    Computes accumulated ΔWEPL along a principal axis.
    """
    spacing_map = {0: spacing_xyz[2], 1: spacing_xyz[1], 2: spacing_xyz[0]}
    voxel_size = spacing_map[axis]
    return np.cumsum(delta_spr, axis=axis) * voxel_size