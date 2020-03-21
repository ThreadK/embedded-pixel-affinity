from __future__ import print_function, division
from typing import Optional, Tuple

import torch
import scipy
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import remove_small_holes
from skimage.measure import label as label_cc # avoid namespace conflict

from data.data_misc import get_padsize, array_unpad

def distance_transform_vol(label, quantize=True, mode='2d'):
    if mode == '3d':
        # calculate 3d distance transform
        vol_distance, vol_semantic = distance_transform(
            label, resolution=(1.0, 1.0, 1.0))
        if quantize:
            vol_distance = energy_quantize(vol_distance)
        return vol_distance

    vol_distance = []
    vol_semantic = []
    for i in range(label.shape[0]):
        label_img = label[i].copy()
        distance, semantic = distance_transform(label_img)
        vol_distance.append(distance)
        vol_semantic.append(semantic)

    vol_distance = np.stack(vol_distance, 0)
    vol_semantic = np.stack(vol_semantic, 0)
    if quantize:
        vol_distance = energy_quantize(vol_distance)
        
    return vol_distance

def distance_transform(label, 
                       bg_value: float = -1.0, 
                       relabel: bool = True, 
                       padding: bool = False,
                       resolution: Tuple[int, ...] = (1.0, 1.0)):
    """Euclidean distance transform