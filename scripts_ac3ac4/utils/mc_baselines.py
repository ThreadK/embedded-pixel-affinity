from concurrent import futures

import numpy as np
import nifty
import nifty.graph.opt.multicut as nmc
import nifty.graph.opt.lifted_multicut as nlmc


# this returns a 2d array with the all the indices of matching rows for a and b
# cf. http://stackoverflow.com/questions/20230384/find-indexes-of-matching-rows-in-two-2-d-arrays
def find_matching_row_indices(x, y):
    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    # using a dictionary, this is faster than the pure np variant
    indices = []
    rows_x = {tuple(row): i for i, row in enumerate(x)}
    for i, row in enumerate(y):
        if tuple(row) in rows_x:
            indices.append([rows_x[tuple(row)], i])
    return np.array(indices)


def compute_mc_superpixels(affinities, n_threads):
    segmenter = McSuperpixel(stacked_2d=True, n_threads=n_threads)
    return segmenter(affinities)


def compute_long_range_mc_superpixels(affinities, offsets,
                                      only_repulsive_lr, n_threads,
                                      stacked_2d=True):
    segmenter = LongRangeMulticutSuperpixel(offsets=offsets, only_repulsive_lr=only_repulsive_lr,
                                            stacke