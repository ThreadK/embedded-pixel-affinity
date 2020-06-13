from __future__ import division

import warnings
import numpy as np

from skimage.morphology import dilation, erosion
from skimage.filters import gaussian

class Compose(object):
    """Composing a list of data transforms. 
    
    The sample size of the composed augmentor can be larger than the 
    specified input size of the model to ensure that all pixels are 
    valid after center-crop.

    Args:
        transforms (list): list of transformations to compose
        input_size (tuple): input size of model in :math:`(z, y, x)` order. Default: :math:`(8, 256, 256)`
        smooth (bool): smoothing the object mask with Gaussian filtering. Default: True
        keep_uncropped (bool): keep uncropped image and label. Default: False
        keep_non_smooth (bool): keep the non-smoothed object mask. Default: False

    Examples::
        >>> augmentor = Compose([Rotate(p=1.0),
        >>>                      Flip(p=1.0),
        >>>                      Elastic(alpha=12.0, p=0.75),
        >>>                      Grayscale(p=0.75),
        >>>                      MissingParts(p=0.9)], 
        >>>                      input_size = (8, 256, 256))
        >>> data = {'image':input, 'label':label}
        >>> augmented = augmentor(data)
        >>> out_input, out_label = augmented['image'], augmented['label']
    """
    def __init__(self, 
                 transforms, 
                 input_size = (8,256,256),
                 smooth = True,
                 keep_uncropped = False,
                 keep_non_smoothed = False):

        self.transforms = transforms
        self.set_flip()

        self.input_size = np.array(input_size)
        self.sample_size = self.input_size.copy()
        self.set_sample_params()

        self.smooth = smooth
        self.keep_uncropped = keep_uncropped
        self.keep_non_smoothed = keep_non_smoothed

    def set_flip(self):
        # Some data augmentation techniques (e.g., elastic wrap, missing parts) are designed only
        # for x-y planes while some (e.g., missing section, mis-alignment) are only applied along
        # the z axis. Thus we let flip augmentation the last one to be applied otherwise shape mis-match
        # can happen when do_ztrans is 1 for cubic input volumes.
        self.flip_aug = None
        flip_idx = None

        for i, t in enumerate(self.transforms):
            if t.__class__.__name__ == 'Flip':
                self.flip_aug = t
                flip_idx = i

        if flip_idx is not None:
            del self.transforms[flip_idx]

    def set_sample_params(self):
        for _, t in enumerate(self.transforms):
            self.sample_size = np.ceil(self.sample_size * t.sample_params['ratio']).astype(int)
            self.sample_size = self.sample_size + (2 * np.array(t.sample_params['add']))
        print('Sample size required for the augmentor:', self.sample_size)

    def smooth_edge(self, data):
        smoothed_label = data['label'].copy()

        for z in range(smoothed_label.shape[0]):
            temp = smoothed_label[z].copy()
            for idx in np.unique(temp):
                if idx != 0:
                    binary = (temp==idx).astype(np.uint8)
                    for _ in range(2):
                        binary = dilation(binary)
                        binary = gaussian(binary, sigma=2, preserve_range=True)
                        binary = dilation(binary)
                        binary = (binary > 0.8).astype(np.uint8)
            
                    temp[np.where(temp==idx)]=0
                    temp[np.where(binary==1)]=idx
            smoothed_label[z] = temp

   