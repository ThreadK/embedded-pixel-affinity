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
        self.softmax = nn.Softmax(dim=1)

        # self.pooling1 = torch.nn.MaxPool3d(kernel_size=(1, 2, 2))
        #
        # self.full2 = self.conv_norm_lrelu(1, self.base_n_filter * 2)
        # self.full3 = self.conv_norm_lrelu(1, self.base_n_filter * 4)
        # self.full4 = self.conv_norm_lrelu(1, self.base_n_filter * 8)
        # self.full5 = self.conv_norm_lrelu(1, self.base_n_filter * 16)
        N = 64
        # stem net
        self.conv1 = nn.Conv3d(1, N, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.IN1 = nn.InstanceNorm3d(N )
        self.conv2 = nn.Conv3d(N, N, kernel_size=3, stride=(1, 2, 2), padding=1,
                               bias=False)
        self.IN2 = n