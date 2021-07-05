
import numpy as np
from .augmentor import DataAugment

from scipy.ndimage.interpolation import map_coordinates, zoom
import numbers
from skimage.draw import line
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import binary_dilation

class MissingParts(DataAugment):
    """Missing-parts augmentation of image stacks.

    Args:
        deformation_strength (int): Default: 0