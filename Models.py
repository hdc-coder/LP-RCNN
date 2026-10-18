# -*- coding: utf-8 -*-
"""Model zoo of the LP-RCNN project.

Every model takes ``[B, C, L, L]`` and returns one score ``[B, 1]``; all share
the two-conv + one-FC backbone of RCNN (Yu et al. 2020):

    CNN            1 ch, 5x5                            -> RCNN
    CNN1           3 ch, 5x5                            -> M-RCNN
    N_RCNN         3 ch, plain RCNN CNN                 -> N-RCNN
    DC_RCNN        1 ch, 5x5                            -> no-multi-scale
    DC_RCNN_MS_DIL 1 ch, dilated 3x3 + global avg pool  -> LP-RCNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """RCNN: 1-channel input, single 5x5 kernel, two conv layers."""

    def __init__(self, L):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.MaxPool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc = nn.Linear(32 * int(L / 4) * int(L / 4), 1)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = self.MaxPool(x)
        x = F.relu(self.conv2(x))
        x = self.MaxPool(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


class CNN1(nn.Module):
    """M-RCNN: 3-channel input (ND / NCom / k-core), single 5x5 kernel."""

    def __init__(self, L):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.MaxPool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc = nn.Linear(32 * int(L / 4) * int(L / 4), 1)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = self.MaxPool(x)
        x = F.relu(self.conv2(x))
        x = self.MaxPool(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


class N_RCNN(nn.Module):
    """N-RCNN (Luo & Lu 2023): 5x5 3->16->32, two max-pools, FC on the flattened map.

    No residual, no batch norm, no global average pooling.
    """

    def __init__(self, L):
        super(N_RCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(32 * int(L / 4) * int(L / 4), 1)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.reshape(x.shape[0], -1)          # flatten, as in the paper
        return self.fc(x)


class DC_RCNN(nn.Module):
    """Single-channel 5x5 baseline with a GAP head, so the ablation isolates multi-scale.
    """

    def __init__(self, L, in_channels=1):
        super(DC_RCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=1, padding=2)
        self.MaxPool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.float()
        x = F.relu(self.conv1(x))
        x = self.MaxPool(x)
        x = F.relu(self.conv2(x))
        x = self.MaxPool(x)
        x = self.avgpool(x).reshape(x.shape[0], -1)   # global average pooling
        return self.fc(x)


class DC_RCNN_MS_DIL(nn.Module):
    """LP-RCNN: dilated multi-scale convolution + global average pooling.

    Three 3x3 branches with dilation 1/2/3 cover 3x3/5x5/7x7 receptive fields at 9
    multiply-adds each; the head pools instead of flattening.
    """

    def __init__(self, L, in_channels=1):
        super(DC_RCNN_MS_DIL, self).__init__()
        self.conv3 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv5 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv7 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=3, dilation=3)
        self.MaxPool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(24, 32, kernel_size=5, stride=1, padding=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        x = x.float()
        b3 = F.relu(self.conv3(x))
        b5 = F.relu(self.conv5(x))
        b7 = F.relu(self.conv7(x))
        x = torch.cat([b3, b5, b7], dim=1)
        x = self.MaxPool(x)
        x = F.relu(self.conv2(x))
        x = self.MaxPool(x)
        x = self.avgpool(x).reshape(x.shape[0], -1)   # global average pooling
        return self.fc(x)


class DC_RCNN_MS_DIL_Flat(nn.Module):
    """LP-RCNN trunk with a flattened head, to isolate global average pooling.
    """

    def __init__(self, L, in_channels=1):
        super(DC_RCNN_MS_DIL_Flat, self).__init__()
        self.conv3 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv5 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv7 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=3, dilation=3)
        self.MaxPool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(24, 32, kernel_size=5, stride=1, padding=2)
        self.fc = nn.Linear(32 * 7 * 7, 1)            # 1568 inputs, no pooling

    def forward(self, x):
        x = x.float()
        b3 = F.relu(self.conv3(x))
        b5 = F.relu(self.conv5(x))
        b7 = F.relu(self.conv7(x))
        x = torch.cat([b3, b5, b7], dim=1)
        x = self.MaxPool(x)
        x = F.relu(self.conv2(x))
        x = self.MaxPool(x)
        x = x.reshape(x.shape[0], -1)                 # flatten the feature map
        return self.fc(x)
