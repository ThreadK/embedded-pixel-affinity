import cv2
import math
import numpy as np
from .augmentor import DataAugment

class MisAlignment(DataAugment):
    """Mis-alignment data augmentation of image stacks.
    
    Args:
        displacement (int): maximum pixel displacement in `xy`-plane. Default: 16
        p (float): probability of applying the augmentation. Default: 0.5
    """
    def __init__(self, 
                 displacement=16, 
                 rotate_ratio=0.0,
                 p=0.5):
        super(MisAlignment, self).__init__(p=p)
        self.displacement = displacement
        self.rotate_ratio = rotate_ratio
        self.set_params()

    def set_params(self):
        self.sample_params['add'] = [0, 
                                     int(math.ceil(self.displacement / 2.0)), 
                                     int(math.ceil(self.displacement / 2.0))]

    def misalignment(self, data, random_state):
        images = data['image'].copy()
        labels = data['label'].copy()

        out_shape = (images.shape[0], 
                     images.shape[1]-self.displacement, 
           