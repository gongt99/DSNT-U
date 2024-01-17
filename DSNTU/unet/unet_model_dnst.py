""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *
import torch
import dsntnn
from dsntnn import dsnt


class UNet_dsnt(nn.Module):
    def __init__(self, n_channels, n_classes, trilinear=False):
        super(UNet_dsnt, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear
        self.inc = DoubleConv(n_channels, 16, 16)
        self.down1 = Down_mod(16, 32)
        self.down2 = Down_mod(32,64)
        self.down3 = Down_mod(64, 128)
        factor = 1
        self.down4 = Down_mod(128, 256 // factor)
        self.up1 = Up(256, 128,trilinear=trilinear)
        self.up2 = Up(128,64,trilinear=trilinear)
        self.up3 = Up(64,32,trilinear=trilinear)
        self.up4 = Up(32,16,trilinear=trilinear)
        #self.up4 = Up(96, 32, trilinear)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        heatmap = dsntnn.flat_softmax(logits)
        coords = dsnt(heatmap, normalized_coordinates=True)
        return coords,heatmap
