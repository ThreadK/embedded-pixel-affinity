import cv2
import math
import random
import numpy as np
from .augmentor import DataAugment

class MotionBlur(DataAugment):
    """Motion blur data augmentation of image stacks.
    
    Args:
        sections (int): number of sections along z dimension to apply motion blur. Default: 2
        kernel_size (int): kernel size for motion blur. Default: 11
        p (float): probability of applying the augmentation. Default: 