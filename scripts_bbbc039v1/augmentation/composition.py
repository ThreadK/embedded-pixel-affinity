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
                 keep_non_s