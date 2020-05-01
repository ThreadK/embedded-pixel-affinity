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
# TODO can we run slic on the full embeddign space without RGB