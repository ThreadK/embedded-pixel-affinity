import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import cv2
import math

__all__ = ['MoE']


class MoE(nn.Module):

    def __init__(self, groups=8, in_channels=512):
        super(MoE, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gap_weight = nn.Sequential(
            nn.Conv2d(in_channels, 8 * self.groups, groups=groups, kernel_size=1, padding=0),
            nn.BatchNorm2d(8 * self.groups),
            nn.Tanh(),
            nn.Conv2d(8 * self.groups, self.groups, groups=groups, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.groups)
        )
        self.bn = nn.BatchNorm2d(self.groups)
        self.sigmoid = nn.Sigmoid()
        self.bn2 = nn.BatchNorm2d(self.groups)

    def forward(self, x):
        # the input is feature map after group conv
        b, c, h, w = x.detach().size()
        x = x.contiguous()
        n = c // self.groups

        # weighted global avg pooling
        pool_weights = self.gap_weight(x)  # b,g,h,w
        pool_weights = pool_weights.view(b, self.groups, -1)
    