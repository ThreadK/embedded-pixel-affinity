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
        >>> augme