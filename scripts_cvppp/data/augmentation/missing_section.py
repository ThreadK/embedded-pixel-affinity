import math
import numpy as np
from .augmentor import DataAugment

class MissingSection(DataAugment):
    """Missing-section augmentation of image stacks.
    
    Args:
        num_sections (int): number of missing sections. Default: 2
        p (float): probability of applying the augmentation. Default: 0.5
    """
    def __init__(self, num_sections=2, p=0.5):
        super(MissingSection, self).__init__(p=p)
        self.num_sections = num_sections
        self.set_params()

    def set_params(self):
        self.sample_params['add'] = [int(math.ceil(self.n