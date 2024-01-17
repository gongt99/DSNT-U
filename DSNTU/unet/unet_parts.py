""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class ResBlock(nn.Module):

    def __init__(self, in_size:int, hidden_size:int, out_size:int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_size, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv3d(hidden_size, out_size, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm3d(hidden_size)
        self.batchnorm2 = nn.BatchNorm3d(out_size)
        self.maxpool = nn.MaxPool3d(2)
    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        return x
    """
    Combine output with the original input
    """
    def forward(self, x):
        return self.maxpool(x + self.convblock(x))

class FC(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.FullConnection = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            #nn.Tanh()
            nn.LeakyReLU()
        )
    def forward(self,x):
        return self.FullConnection(x)

class Down(nn.Module):

    def __init__(self, in_channels,out_channels,mid_channels=None):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels,out_channels, mid_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Down_mod(nn.Module):

    def __init__(self, in_channels,out_channels,mid_channels=None):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            DoubleConv(in_channels,out_channels, mid_channels),
            nn.MaxPool3d(2)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None,trilinear=True):
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels )
        else:
            self.up = nn.ConvTranspose3d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2,
                        diffX // 2, diffX - diffX // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class Up_mod(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, mid_channels=None,trilinear=True):
        super().__init__()

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels )
        else:
            self.up = DoubleConv(in_channels, out_channels)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffY // 2, diffY - diffY // 2,
                        diffX // 2, diffX - diffX // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Channel_Attention(nn.Module):
    def __init__(self,channels,r=16):
        super(Channel_Attention,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Conv3d(channels, channels // r, kernel_size=1, bias=False),
            nn.ReLU(True),
            nn.Conv3d(channels // r, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        x1 = self.avg_pool(x)
        x1 = self.fc(x1)
        x2 = self.max_pool(x)
        x2 = self.fc(x2)

        y = self.sigmoid(x1+x2)
        return x * y

class Spatial_Attention(nn.Module):
    def __init__(self):
        super(Spatial_Attention,self).__init__()
        padding = 3
        self.conv = nn.Conv3d(2,1,kernel_size=7,stride=1,padding=padding,bias=False)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout,_ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avgout,maxout], dim=1)
        x2 = self.conv(x1)
        return self.sigmoid(x2)*x

class Down_CBAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )
        self.channel_attention = Channel_Attention(in_channels)
        self.spatial_attention = Spatial_Attention()

    def forward(self, x):
        x = self.spatial_attention(x)
        x = self.channel_attention(x)
        return self.maxpool_conv(x)




class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
