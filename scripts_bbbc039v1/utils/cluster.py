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
    bg_mask = seg == 0
    # don't need to  do anything if we only have background
    if bg_mask.size == int(bg_mask.sum()):
        return seg

    hmap = vigra.filters.gaussianSmoothing(raw, sigma=1.)

    bg_mask = binary_erosion(bg_mask, iterations=erosion_iters)
    seg_new = seg.copy()
    bg_id = int(seg.max()) + 1
    seg_new[bg_mask] = bg_id

    seg_new, _ = vigra.analysis.watershedsNew(hmap, seeds=seg_new.astype('uint32'))

    seg_new[seg_new == bg_id] = 0
    return seg_new


def cluster(emb, clustering_alg, semantic_mask=None):
    output_shape = emb.shape[1:]
    # reshape (E, D, H, W) -> (E, D * H * W) and transpose -> (D * H * W, E)
    flattened_embeddings = emb.reshape(emb.sha