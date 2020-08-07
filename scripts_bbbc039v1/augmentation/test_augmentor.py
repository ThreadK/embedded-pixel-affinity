import numpy as np
import itertools
import torch

class TestAugmentor(object):
    """Test-time augmentor. 
    
    Our test-time augmentation includes horizontal/vertical flips 
    over the `xy`-plane, swap of `x` and `y` axes, and flip in `z`-dimension, 
    resulting in 16 variants. Considering inference efficiency, we also 
    provide the option to apply only `x-y` swap and `z`-flip, resulting in 4 variants.
    By default the test-time augmentor returns the pixel-wise mean value of the predictions.

    Args:
        mode (str): one of ``'min'``, ``'max'`` or ``'mean'``. Default: ``'mean'``
        num_aug (int): number of data augmentation variants: 0, 4 or 16. Default: 4

    Examples::
        >>> from connectomics.data.augmentation import TestAugmentor
        >>> test_augmentor = TestAugmentor(mode='mean', num_aug=16)
        >>> output = test_augmentor(model, inputs) # output is a numpy.ndarray on CPU
    """
    def __init__(self, mode='mean', num_aug=4):
        self.mode = mode
        self.num_aug = num_aug
        assert num_aug in [0, 4, 16], "TestAugme