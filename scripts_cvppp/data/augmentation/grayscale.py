import numpy as np
from .augmentor import DataAugment

class Grayscale(DataAugment):
    """Grayscale intensity augmentation, adapted from ELEKTRONN (http://elektronn.org/).

    Randomly adjust contrast/brightness, randomly invert the color space
    and apply gamma correction.

    Args:
        contrast_factor (float): intensity of contrast change. Default: 0.3
        brightness_factor (float): intensity of brightness change. Default: 0.3
        mode (string): one of ``'2D'``, ``'3D'`` or ``'mix'``. Default: ``'mix'``
        p (float): probability of applying the augmentation. Default: 0.5
    """

    def __init__(self, contrast_factor=0.3, brightness_factor=0.3, mode='mix', p=0.5):
        """Initialize parameters.
        "