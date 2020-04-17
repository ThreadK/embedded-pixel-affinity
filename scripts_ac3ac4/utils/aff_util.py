import numpy as np
#from em_segLib.seg_util import check_volume
# from scipy.misc import comb
from scipy.special import comb
import scipy.sparse


def affinitize(img, ret=None, dst=(1,1,1), dtype='float32'):
    # PNI code
    """
    Transform segment