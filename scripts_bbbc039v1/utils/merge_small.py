import numpy as np
import vigra
from concurrent import futures
# from Tools import UnionFind

class Node(object):
    def __init__(self, u):
        self.parent = self
        self.label  = u
        self.rank   = 0

class UnionFind(object):

    def __init__(self, n_labels):
        assert isinstance(n_labels, int), type(n_labels)
        self.n_labels = n_labels
        self.nodes = [Node(n) for n in range(n_labels)]


    # find the root of u and compress the path on the way
    def find(self, u_id):
        #assert u_id < self.n_labels
        u = self.nodes[ u_id ]
        return self.findNode(u)

    # find the root of u and compress the path on the way
    def findNode(self, u):
        if u.parent == u:
            return u
        else:
            u.parent = self.findNode(u.parent)
            return u.parent

    def merge(self, u_id, v_id):
        #assert u_id < self.n_labels
        #assert v_id < self.n_labels
        u = self.nodes[ u_id ]
        v = self.nodes[ v_id ]
        self.mergeNode(u, v)

    # merge u and v trees in a union by rank manner
    def mergeNode(self, u, v):
        u_root = self.findNode(u)
        v_root = self.findNode(v)
        if u_root.rank > v_root.rank:
            v_root.parent = u_root
        elif u_root.rank < v_root.rank:
            u_root.parent = v_root
        elif u_root != v_root:
            v_root.parent = u_root
            u_root.rank += 1

    # get the new sets after merging
    def get_merge_result(self):

        merge_result = []

        # find all the unique roots
        roots = []
        for u in self.nodes:
            root = self.findNode(u)
            if not root in roots:
                roots.append(root)

        # find ordering of roots (from 1 to n_roots)
        roots_ordered = {}
        root_id = 0
        for root in roots:
            merge_result.append( [] )
            roots_ordered[root] = root_id
            root_id += 1
        for u in self.nodes:
            u_label = u.label
            root = self.findNode(u)
            merge_result[ roots_ordered[root] ].append(u_label)

        # sort the nodes in the result
        #(this might result in problems if label_type cannot be sorted)
        for res in merge_result:
            res.sort()

        return merge_result


# numpy.replace: replcaces the values in array according to dict
# cf. SO: http://stackoverflow.com/questions/3403973/fast-replacement-of-values-in-a-numpy-array
def replace_from_dict(array, dict_like):
    replace_keys, replace_vals = np.array(list(zip( *sorted(dict_like.items() ))))
    # FIXME This is just some dirty hack because I can't get np version 1.10 to run
    if np.__version__ == '1.9.3':
        indices = np.digitize(array.flatten(), replace_keys, right=True)
        return replace_vals[indices].astype(array.dtype).reshape(array.shape)
    else:
        indices = np.digitize(array, replace_keys, right = True)
        return replace_vals[indices].astype(array.dtype)

# TODO 10,000 seems to be a pretty large default value !
# TODO FIXME rethink the relabeling here, in which cases do we want it, can it hurt?
def remove_small_segments(segmentation,
        size_thresh = 10000,
        relabel = True):

    # Make sure all objects have their individual label
    # TODO FIXME this is very dangerous for sample C (black slices !)!
    if relabel:
        segmentation = vigra.analysis.labelVolumeWithBackground(
            segmentation.astype('uint32'), neighborhood=6, background_value=0)

    # Get the unique values of the segmentation including counts
    uniq, counts = np.unique(segmentation, return_counts=True)

    # Keep all uniques that have a count smaller than size_thresh
    small_objs = uniq[counts < size_thresh]
    large_objs = uniq[counts >= size_thresh]
    print('len(large_objs) == {}'.format(len(large_objs)))
    print('len(small_objs)