import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from .augmentor import DataAugment

class Elastic(DataAugment):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    The implementation is based on https: