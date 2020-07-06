import cv2
import math
import numpy as np
from .augmentor import DataAugment

class MisAlignment(DataAugment):
    """Mis-alignment data augmentation of image stacks.
    
    Args:
        displacement (int): maximum pixel displacement in `xy`-plane. Default: 16
        p (float): probability of applying the augmentation. Default: 0.5
    """
    def __init__(self, 
                 displacement