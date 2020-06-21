import numpy as np
from .augmentor import DataAugment

class Grayscale(DataAugment):
    """Grayscale intensity augmentation, adapted from ELEKTRONN (http://elektronn.org/).

    Randomly adjust contrast/brightness, randomly invert the color space
    