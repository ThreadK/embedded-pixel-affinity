import numpy as np
from scipy.sparse import coo_matrix
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from skimage.morphology import erosion, dilation
from skimage.measure import label as label_cc # avoid namespace conflict
from skimage.segmentation import find_boundaries

from data.data_affinity import mknhood2d, seg_to_aff
from data.data_transform import distance_transform_vol

# reduce the labeling
def getSegType(mid):
    m_type = np.uint64
    if mid<2**8:
        m_type = np.uint8
    elif mid<2**16:
        m_type = np.uint16
    elif mid<2**32:
        m_type = np.uint32
    return m_type

def relabel(seg, do_type=False):
    # get the unique labels
    uid = np.unique(seg)
    # ignore all-b