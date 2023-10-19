import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from functools import partial
from einops import rearrange


class Block_plus(nn.Module):
    def __init__(self, in_ch, middle_ch, out_ch):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, middle_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_ch)
        self.conv2 = nn.Conv2d(middle_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        return out


class UpPlus(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)

        return x