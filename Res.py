import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from functools import partial
from einops import rearrange


class ResBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            #nn.InstanceNorm2d(out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            #nn.LeakyReLU(inplace=True),
            # nn.SELU(inplace=True),
            # nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            #nn.InstanceNorm2d(out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            #nn.LeakyReLU(inplace=True)
            # nn.SELU(inplace=True)

        )

    def forward(self, x):
        residual = x

        out = self.conv(x)

        out = out + residual

        return out


class ResBlock_noPadding(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(ResBlock_noPadding, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(inplace=True),
            # nn.SELU(inplace=True),
            # nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            # nn.InstanceNorm2d(out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
            # nn.LeakyReLU(inplace=True)
            # nn.SELU(inplace=True)

        )

    def forward(self, x):
        residual = x

        out = self.conv(x)

        out = out + residual

        out = out[:, :, 2:out.shape[-2] - 2, 2:out.shape[-1] - 2]

        return out


class stadge_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(stadge_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            # nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2),
            # nn.Conv2d(in_ch, out_ch, kernel_size=3),
            nn.InstanceNorm2d(out_ch),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            # nn.SELU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            # nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=2, dilation=2),
            # nn.Conv2d(out_ch, out_ch, kernel_size=3),
            nn.InstanceNorm2d(out_ch),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=True)
            # nn.SELU(inplace=True)
         )

    def forward(self, x):
        x = self.conv(x)
        return x


class DeFusion(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DeFusion, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            # nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=2, dilation=2),
            # nn.Conv2d(in_ch, out_ch, kernel_size=3),
            nn.InstanceNorm2d(out_ch),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(inplace=True),
            # nn.SELU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            # nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=2, dilation=2),
            # nn.Conv2d(out_ch, out_ch, kernel_size=3),
            nn.InstanceNorm2d(out_ch),
            # nn.BatchNorm2d(out_ch),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=True)
            # nn.SELU(inplace=True)
        )

    def forward(self, x1, x2):
        '''
        if x1.shape[2] != x2.shape[2]:
            startIndexDim2 = (x2.size()[2] - x1.size()[2]) // 2
            startIndexDim3 = (x2.size()[3] - x1.size()[3]) // 2
            x = torch.cat(
                [x2[:, :, startIndexDim2:x1.size()[2] + startIndexDim2, startIndexDim3:x1.size()[3] + startIndexDim3],
                 x1], dim=1)
        else:
            x = torch.cat([x2, x1], dim=1)
        '''

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x