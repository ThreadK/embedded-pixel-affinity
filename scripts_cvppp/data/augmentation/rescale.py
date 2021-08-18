from __future__ import division
import cv2
import numpy as np
from .augmentor import DataAugment
from skimage.transform import resize

class Rescale(DataAugment):
    """
    Rescale augmentation.
    
    Args:
        low (float): lower bound of the random scale factor. Default: 0.8
        high (float): higher bound of the random scale factor. Default: 1.2
        fix_aspect (bool): fix aspect ratio or not. Default: False
        p (float): probability of applying the augmentation. Default: 0.5
    """
    def __init__(self, low=0.8, high=1.2, fix_aspect=False, p=0.5):
        super(Rescale, self).__init__(p=p) 
        self.low = low
        self.high = high
        self.fix_aspect = fix_aspect

        self.image_interpolation = 1
        self.label_interpolation = 0
        self.set_params()

    def set_params(self):
        assert (self.low >= 0.5)
        assert (self.low <= 1.0)
        ratio = 1.0 / self.low
        self.sample_params['ratio'] = [1.0, ratio, ratio]

    def random_scale(self, random_state):
        rand_scale = random_state.rand() * (self.high - self.low) + self.low
        return rand_scale

    def apply_rescale(self, image, label, sf_x, sf_y, random_state):
        # apply image and mask at the same time
        transformed_image = image.copy()
        transformed_label = label.copy()

        y_length = int(sf_y * image.shape[1])
        if y_length <= image.shape[1