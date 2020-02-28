import numpy as np

class DataAugment(object):
    """
    DataAugment interface.

    A data transform needs to conduct the following steps:

    1. Set :attr:`sample_params` at initialization to compute required sample size.
    2. Randomly generate augmentation parameters for the current transform.
    3. Apply the transform to a pair of images and corresponding labels.

    All the real data augmentations should be a subclass of this class.
    """
    def __init__(self, p=0.5):
        assert p >= 0.0 and p <=1.0
        self.p = p
 