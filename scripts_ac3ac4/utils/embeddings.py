import numpy as np

from scipy.ndimage import shift
from sklearn.decomposition import PCA

from skimage.segmentation import slic


def embedding_pca(e