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

    def __init__(self, contrast_factor=0.3, brightness_factor=0.3, mode='3D', p=0.5):
        """Initialize parameters.
        """
        super(Grayscale, self).__init__(p=p)
        self._set_mode(mode)
        self.CONTRAST_FACTOR   = contrast_factor
        self.BRIGHTNESS_FACTOR = brightness_factor

    def set_params(self):
        # No change in sample size
        pass

    def __call__(self, data, random_state=np.random):

        if self.mode == 'mix':
            mode = '3D' if random_state.rand() > 0.5 else '2D'
        else:
            mode = self.mode

        # apply augmentations  
        if mode == '2D': 
            data = self._augment2D(data, random_state)
        if mode == '3D': 
            data = self._augment3D(data, random_state)
        return data

    def _augment2D(self, data, random_state=np.random):
        """
        Adapted from ELEKTRONN (http://elektronn.org/).
        """
        imgs = data['image']
        transformedimgs