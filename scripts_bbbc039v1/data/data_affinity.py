import numpy as np

# from Janelia pyGreentea
# https://github.com/naibaf7/PyGreentea
def mknhood2d(radius=1):
    # Makes nhood structures for some most used dense graphs.

    ceilrad = np.ceil(radius)
    x = np.arange(-ceilrad,ceilrad+1,1)
    y = np.arange(-ceilrad,ceilrad+1,1)
    [i,j] = np.meshgrid(y,x)

    idxkeep = (i**2+j**2)<=radius**2
    i=i[idxkeep].ravel(); j=j[idxkeep].ravel();
    zeroIdx = np.ceil(len(i)/2).astype(np.int32);

    nhood = np.vstack((i[:zeroIdx],j[:zeroIdx])).T.astype(np.int32)
    nhood = np.ascontiguousarray(np.flipud(nhood))
    nhood = nhood[1:]
    return nhood 

def mknhood3d(radius=1):
    # Makes nhood structures for some most used dense graphs.
    #