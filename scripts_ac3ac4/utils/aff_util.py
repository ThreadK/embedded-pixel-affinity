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
            s0.append(slice(dst[i],  None))
            s1.append(slice(dst[i],  None))
            s2.append(slice(None, -dst[i]))
        else:
            s0.append(slice(None,  dst[i]))
            s1.append(slice(-dst[i], None))
            s2.append(slice(None,  dst[i]))

    ret[s0] = (img[s1]==img[s2]) & (img[s1]>0)
    return ret[np.newaxis,...]

def bmap_to_affgraph(bmap,nhood,return_min_idx=False):
    # constructs an affinity graph from a boundary map
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = bmap.shape
    nEdge = nhood.shape[0]
    aff = np.zeros((nEdge,)+shape,dtype=np.int32)
    minidx = np.zeros((nEdge,)+shape,dtype=np.int32)

    for e in range(nEdge):
        aff[e, \
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] = np.minimum( \
                        bmap[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])], \
                        bmap[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                          