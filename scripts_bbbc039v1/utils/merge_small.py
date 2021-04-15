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


    # find the root of u and com