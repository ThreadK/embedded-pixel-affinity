# import hdbscan
import numpy as np
# import vigra
from numpy import linalg as LA
from scipy.ndimage import binary_erosion
from sklearn.cluster import DBSCAN, MeanShift


def iou(gt, seg):
    epsilon = 1e-5
    inter = (gt & seg).sum()
    union = (gt | seg).sum()

    iou = (inter + epsilon) / (union + epsilon)
    return iou


def expand_labels_watershed(seg, raw, erosion_iters=4):
    b