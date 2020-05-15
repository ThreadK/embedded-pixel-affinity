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
    # using a dictionary, this is faster than the pure np varian