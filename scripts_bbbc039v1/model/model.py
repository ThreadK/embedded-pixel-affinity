import torch
import torch.nn as nn
import torch.nn.functional as F

class Modified3DUNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=2, base_n_filter=64):
        super(Modified3DUNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.6)
        self.upsacle = nn.Upsample(scale_factor=2, mode='trilinear',align_corners=False)
        self.softmax = nn.Softmax