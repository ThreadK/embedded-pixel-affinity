import numpy as np
#from em_segLib.seg_util import check_volume
# from scipy.misc import comb
from scipy.special import comb
import scipy.sparse


def affinitize(img, ret=None, dst=(1,1,1), dtype='float32'):
    # PNI code
    """
    Transform segmentation to an affinity map.
    Args:
        img: 3D indexed image, with each index corresponding to each segment.
    Returns:
        ret: an affinity map (4D tensor).
    """
    img = check_volume(img)
    if ret is None:
        ret = np.zeros(img.shape, dtype=dtype)

    # Sanity check.
    (dz,dy,dx) = dst
    assert abs(dx) < img.shape[-1]
    assert abs(dy) < img.shape[-2]
    assert abs(dz) < img.shape[-3]

    # Slices.
    s0 = list()
    s1 = list()
    s2 = list()
    for i in range(3):
        if dst[i] == 0:
            s0.append(slice(None))
            s1.append(slice(None))
            s2.append(slice(None))
        elif dst[i] > 0:
            s0.append(slic