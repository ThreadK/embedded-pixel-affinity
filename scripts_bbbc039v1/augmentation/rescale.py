from __future__ import division
import cv2
import numpy as np
from .augmentor import DataAugment
from skimage.transform import resize

class Rescale(DataAugment):
    """
    Rescale augmentation.
    
    Args:
        low (float): lower bound of the random scale factor. Default: 0.8
        high (float): higher bound of the random scale factor. Default: 1.2
        fix_aspect (bool): fix aspect ratio or not. Default: False
        p (float): probability of applying the augmentation. Default: 0.5
    """
    def __init__(se