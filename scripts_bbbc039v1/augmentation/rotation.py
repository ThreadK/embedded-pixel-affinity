import cv2
import numpy as np
from .augmentor import DataAugment

class Rotate(DataAugment):
    """
    Continuous rotatation of the `xy`-plane.

    The sample size for `x`- and `y`-axes should be at least :math: