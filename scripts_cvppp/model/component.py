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
            nn.Conv2d(in_channels, 8 * self.groups, groups=groups, kern