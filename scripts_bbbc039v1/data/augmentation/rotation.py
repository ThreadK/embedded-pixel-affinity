import cv2
import numpy as np
from .augmentor import DataAugment

class Rotate(DataAugment):
    """
    Continuous rotatation of the `xy`-plane.

    The sample size for `x`- and `y`-axes should be at least :math:`\sqrt{2}` times larger
    than the input size to make sure there is no non-valid region after center-crop.
    
    Args:
        p (float): probability of applying the augmentation. Default: 0.5
    """
    def __init__(self, p=0.5):
        super(Rotate, self).__init__(p=p) 
        self.image_interpolation = cv2.INTER_LINEAR
        self.label_interpolation = cv2.INTER_NEAREST
        self.border_mode = cv2.BORDER_CONSTANT
        self.s