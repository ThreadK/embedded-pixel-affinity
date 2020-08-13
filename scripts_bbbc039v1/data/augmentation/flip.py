import random
import numpy as np
from .augmentor import DataAugment

class Flip(DataAugment):
    """
    Randomly flip along `z`-, `y`- and `x`-axes as well as swap `y`- and `x`-axes 
    for anisotropic image volumes. For learning on is