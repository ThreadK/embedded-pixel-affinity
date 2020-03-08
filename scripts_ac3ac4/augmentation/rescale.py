from __future__ import division
import cv2
import numpy as np
from .augmentor import DataAugment
from skimage.transform import resize

class Rescale(DataAugment):
    """
 