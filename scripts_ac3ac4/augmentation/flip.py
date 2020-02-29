import random
import numpy as np
from .augmentor import DataAugment

class Flip(DataAugment):
    """
    Randomly flip along `z`-, `y`- and `x`-axes as well as swap `y`- and `x`-axes 
    for anisotropic image volumes. For learning on isotropic image volumes set 
    :attr:`do_ztrans` to 1 to swap `z`- and `x`-axes (the inputs need to be cubic).

    Args:
        p (float): probability of applying the augmentation. Default: 0.5
        do_ztrans (int): set to 1 to swap z- and x-axes for isotropic data. Default: 0
    """
    def __init__(self, p=0.5, do_ztrans=0):
        super(Flip, self).__in