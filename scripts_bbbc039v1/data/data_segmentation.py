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
    # ignore all-background samples
    if len(uid)==1 and uid[0] == 0:
        return seg

    uid = uid[uid > 0]
    mid = int(uid.max()) + 1 # get the maximum label for the segment

    # create an array from original segment id to reduced id
    m_type = seg.dtype
    if do_type:
        m_type = getSegType(mid)
    mapping = np.zeros(mid, dtype=m_type)
    mapping[uid] = np.arange(1, len(uid) + 1, dtype=m_type)
    return mapping[seg]

def remove_small(seg, thres=100):
    sz = seg.shape
    seg = seg.reshape(-1)
    uid, uc = np.unique(seg, return_counts=True)
    seg[np.in1d(seg,uid[uc<thres])] = 0
    return seg.reshape(sz)

def im2col(A, BSZ, stepsize=1):
    # Parameters
    M,N = A.shape
    # Get Starting block indices
    start_idx = np.arange(0,M-BSZ[0]+1,stepsize)[:,None]*N + np.arange(0,N-BSZ[1]+1,stepsize)
    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(BSZ[0])[:,None]*N + np.arange(BSZ[1])
    # Get all actual indices & index into input array for final output
    return np.take(A,start_idx.ravel()[:,None] + offset_idx.ravel())

def seg_widen_border(seg, tsz_h=1):
    # Kisuk Lee's thesis (A.1.4): 
    # "we preprocessed the ground truth seg such that any voxel centered on a 3 × 3 × 1 window containing 
    # more than one positive segment ID (zero is reserved for background) is marked as background."
    # seg=0: background
    tsz = 2*tsz_h+1
    sz = seg.shape
    if len(sz)==3:
        for z in range(sz[0]):
            mm = seg[z].max()
            patch = im2col(np.pad(seg[z],((tsz_h,tsz_h),(tsz_h,tsz_h)),'reflect'),[tsz,tsz])
            p0=patch.max(axis=1)
            patch[patch==0] = mm+1
            p1=patch.min(axis=1)
            seg[z] =seg[z]*((p0==p1).reshape(sz[1:]))
    else:
        mm = seg.max()
        patch = im2col(np.pad(seg,((tsz_h,tsz_h),(tsz_h,tsz_h)),'reflect'),[tsz,tsz])
        p0 = patch.max(axis=1)
        patch[patch == 0] = mm + 1
        p1 = patch.min(axis = 1)
        seg = seg * ((p0 == p1).reshape(sz))
    return seg

def seg_to_small_seg(seg,thres=25,rr=2):
    # rr: z/x-y resolution ratio
    sz = seg.shape
    mask = np.zeros(sz,np.uint8)
    for z in np.where(seg.max(axis=1).max(axis=1)>0)[0]:
        tmp = label_cc(seg[z])
        ui,uc = np.unique(tmp,return_counts=True)
        rl = np.zeros(ui[-1]+1,np.uint8)
        rl[ui[uc<thres]]=1;rl[0]=0
        mask[z] += rl[tmp]
    for y in np.where(seg.max(axis=2).max(axis=0)>0)[0]:
        tmp = label_cc(seg[:,y])
        ui,uc = np.unique(tmp,return_counts=True)
        rl = np.zeros(ui[-1]+1,np.uint8)
        rl[ui[uc<thres//rr]]=1;rl[0]=0
        mask[:,y] += rl[tmp]
    for x in np.where(seg.max(axis=0).max(axis=0)>0)[0]:
        tmp = label_cc(seg[:,:,x])
        ui,uc = np.unique(tmp,return_counts=True)
        rl = np.zeros(ui[-1]+1,np.uint8)
        rl[ui[uc<thres//rr]]=1;rl[0]=0
        mask[:,:,x] += rl[tmp]
    return mask

def seg_to_instance_bd(seg, tsz_h=7, do_bg=False):
    tsz = tsz_h*2+1
    mm = seg.max()
    sz = seg.shape
    bd = np.zeros(sz, np.uint8)
    for z in range(sz[0]):
        patch = im2col(np.pad(seg[z], ((tsz_h,tsz_h),(tsz_h,tsz_h)),'reflect'),[tsz,tsz])
        p0 = patch.max(axis=1)
        if do_bg: # at least one non-zero seg
            p1 = patch.min(axis=1)
            bd[z] = ((p0>0)*(p0!=p1)).reshape(sz[1:])
        else: # between two non-zero seg
            patch[patch==0] = mm+1
            p1 = patch.min(axis=1)
            bd[z] = ((p0!=0)*(p1!=0)*(p0!=p1)).reshape(sz[1:])
    return bd

def markInvalid(seg, iter_num=2, do_2d=True):
    # find invalid 
    # if do erosion(seg==0), then miss the border
    if do_2d:
        stel=np.array([[1,1,1], [1,1,1]]).astype(bool)
        if len(seg.shape)==2:
            out = binary_dilation(seg>0, structure=stel, iterations=iter_num)
            seg[out==0] = -1
        else: # save memory
            for z in range(seg.shape[0]):
                tmp = seg[z] # by reference
                out = binary_dilation(tmp>0, structure=stel, iterations=iter_num)
                tmp[out==0] = -1
    else:
        stel=np.array([[1,1,1], [1,1,1], [1,1,1]]).astype(bool)
        out = binary_dilation(seg>0, structure=stel, iterations=iter_num)
        seg[out==0] = -1
    return seg

def seg_to_weights(targets, wopts, mask=None):
    # input: list of targets
    out=[None]*len(wopts)
    for wid, wopt in enumerate(wopts):
        # 0: no weight
        out[wid] = seg_to_weight(targets[wid], wopt, mask)
    return out

def seg_to_weight(target, wopts, mask=None):
    out=[None]*len(wopts)
    foo = np.zeros((1), int)
    for wid, wopt in enumerate(wopts):
        # 0: no weight
        out[wid] = foo
        if wopt == '1': # 1: by gt-target ratio 
            out[wid] = weight_binary_ratio(target, mask)
        elif wopt == '2': # 2: unet weight
            out[wid] = weight_unet3d(target)
    return out

def seg_to_targets(label, topts):
    # input: (D, H, W)
    # output: (C, D, H, W)
    out = [None]*len(topts)
    for tid, topt in enumerate(topts):
        if topt[0] == '9': # generic segmantic segmentation
            out[tid] = label.astype(np.int64)
        elif topt == '0': # binary
            out[tid] = (label>0)[None,:].astype(np.float32)
        elif topt[0] == '1': # synaptic polarity:
            tmp = [None]*3 
            tmp[0] = np.logical_and((label % 2) == 1, label > 0)
            tmp[1] = np.logical_and((label % 2) == 0, label > 0)
            tmp[2] = (label > 0)
            out[tid] = np.stack(tmp, 0).astype(np.float32)
        elif topt[0] == '2': # affinity
            if label.ndim == 3: # 3d aff 
                out[tid] = seg_to_aff(label)
            elif label.ndim == 2: # 2d aff 
                out[tid] = seg_to_aff(label, nhood=mknhood2d(1))
            else:
                raise ValueError('Undefined affinity computation for ndim = ' + str(label.ndim))
        elif topt[0] == '3': # small object mask
            # size_thres: 2d threshold for small size
            # zratio: resolution ration between z and x/y
            # mask_dsize: mask dilation size
            _, size_thres, zratio, _ = [int(x) for x in topt.split('-')]
            out[tid] = (seg_to_small_seg(label, size_thres, zratio)>0)[None,:].astype(np.float32)
        elif topt[0] == '4': # instance