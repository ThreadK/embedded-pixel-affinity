
import os
import numpy as np
from PIL import Image
from dataset.transforms import RandomAffine
from dataset.dataset import MyDataset
import glob


class LeavesDataset(MyDataset):

    def __init__(self,
                 leaves_dir='',
                 leaves_test_dir='',
                 batch_size=1,
                 gt_maxseqlen=20,
                 transform=None,
                 target_transform=None,
                 augment=False,
                 split='train',
                 resize=False,
                 imsize=256,
                 rotation=10,
                 translation=0.1,
                 shear=0.1,
                 zoom=0.7,
                 num_train=108,
                 padding=True):

        CLASSES = ['<eos>', 'leaf']

        self.split = split
        self.num_train = num_train
        self.padding = padding
        self.classes = CLASSES
        self.num_classes = len(self.classes)
        self.max_seq_len = gt_maxseqlen