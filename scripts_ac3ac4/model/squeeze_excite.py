import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from em_net.libs.sync import SynchronizedBatchNorm1d, SynchronizedBatchNorm3d

# -- squeeze-and-excitation layer --
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                SynchronizedBatchNorm1d(channel // reduction),
                nn.ELU(inplace=True),
                nn.Linear(channel // reduction, channel),
                SynchronizedBatchNorm1d(channel),
                nn.Sigmoid())

    def forward(self, x):
   