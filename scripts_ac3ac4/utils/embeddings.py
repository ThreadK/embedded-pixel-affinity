import numpy as np

from scipy.ndimage import shift
from sklearn.decomposition import PCA

from skimage.segmentation import slic


def embedding_pca(embeddings, n_components=3, as_rgb=True):
    """
    """

    if as_rgb and n_components != 3:
        raise ValueError("")

    pca = PCA(n_components=n_components)
    embed_dim = embeddings.shape[0]
    shape = embeddings.shape[1:]

    embed_flat = embeddings.reshape(embed_dim, -1).T
    embed_flat = pca.fit_transform(embed_flat).T
    embed_flat = embed_flat.reshape((n_components,) + shape)

    if as_rgb:
        embed_flat = 255 * (embed_flat - embed_flat.min()) / np.ptp(embed_flat)
        embed_flat = embed_flat.astype('uint8')

    return embed_flat


# TODO which slic implementation to use? vigra or skimage?
# TODO can we run slic on the full embeddign space without RGB PCA?
# TODO is there a slic implementation that can do this?
# TODO slic parameter?
def embedding_slic(embeddings, run_pca=True):
    """
    """

    if run_pca:
        embeddings = embedding_pca(embeddings, as_rgb=True)
    embeddings = embeddings.transpose((1, 2, 0)) if embeddings.ndim == 3 else\
        embeddings.transpose((1, 2, 3, 0))
    seg = slic(embeddings, convert2lab=True)
    # print(embeddings.shape)
    # seg = slicSuperpixels(embeddings[..., 0], intensityScaling=1., seedDistance=1)[0]
    return seg


def _embeddings_to_probabilities(embed1, embed2, delta, embedding_axis):
    probs = (2 * delta - np.linalg.norm(embed1 - embed2, axis=embedding_axis)) / (2 * delta)
   