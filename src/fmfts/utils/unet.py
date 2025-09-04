import copy
import torch
import torch.nn as nn 
from fmfts.utils import padding

class MaxPool(nn.Module):
    def __init__(self, *ks):
        super().__init__()
        if len(ks) == 1:   self.mp = nn.MaxPool1d(ks, ks)
        elif len(ks) == 2: self.mp = nn.MaxPool2d(ks, ks)
        elif len(ks) == 3: self.mp = nn.MaxPool3d(ks, ks)
        else: raise Exception()

    def forward(self, x):
        return self.mp(x)

class DoubleConv(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 padding="same", 
                 padding_mode="zeros", 
                 dims=2,
                 nl=nn.ReLU()):
        super().__init__()
        cls = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dims-1]
        self.nl = nl 
        self.conv1 = cls(in_channels,  out_channels, 3, padding=padding, padding_mode=padding_mode)
        self.conv2 = cls(out_channels, out_channels, 3, padding=padding, padding_mode=padding_mode)

    def forward(self, x):
        x = self.conv1(x)
        x = self.nl(x)
        x = self.conv2(x)
        return x

class ResNetBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 padding,
                 dims=2,
                 nl=nn.ReLU()):
        super().__init__()
        cls = [nn.Conv1d, nn.Conv2d, nn.Conv3d][dims-1]
        self.dc = DoubleConv(in_channels, out_channels, padding="valid", dims=dims, nl=nl)
        self.conv = cls(in_channels, out_channels, kernel_size=1)
        self.padding = padding
        self.dims = dims

    def forward(self, x):
        x1, x2 = x.clone(), x.clone()
        for pm, dim in zip(self.padding, range(self.dims)):
            x1 = padding.pad(x1, dim=2+dim, extent=2, padding_mode=pm)
        y = self.conv(x2)
        z = self.dc(x1)
        return y + z

class UNet(nn.Module):
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1, 
                 features=(32, 64),
                 padding=("zeros", "zeros"),
                 nl=nn.ReLU()):
        super().__init__()
        dims = len(padding)
        assert dims in [1,2,3]
        
        self.pool = MaxPool(*[2]*dims)
        self.nl = nl
        self.first = ResNetBlock(in_channels, features[0], padding=padding, nl=self.nl, dims=dims)
        self.encoders = nn.ModuleList([
                ResNetBlock(
                    features[k],
                    features[k+1],
                    padding=padding,
                    nl=self.nl,
                    dims=dims
                ) 
                for k in range(len(features)-1)])
        self.bottleneck = ResNetBlock(
            features[-1],
            features[-1],
            padding=padding,
            nl = self.nl, 
            dims=dims)
        self.decoders = nn.ModuleList([
            ResNetBlock(
                2*features[k-1],
                features[k-1],
                padding=padding,
                nl=self.nl,
                dims=dims) 
            for k in range(len(features)-1, 0, -1)])
        self.up = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=tuple([2]*dims), mode="nearest"), # <- "bilinear" doesn't support 3d
                ResNetBlock(
                    features[k],
                    features[k-1],
                    padding=padding,
                    nl=self.nl,
                    dims=dims)
            ) for k in range(len(features)-1, 0, -1)])
        self.final = ResNetBlock(features[0], out_channels, padding=padding, nl=self.nl, dims=dims)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = self.first(x)
        encoded = []
        for encoder in self.encoders:
            encoded.append(x)
            x = encoder(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for decoder, up, encx in zip(self.decoders, self.up, encoded[::-1]):
            x = up(x)
            x = torch.cat([x, encx], dim=1)
            x = self.nl(decoder(x))
        x = self.final(x)
        return x
    
    def clone_and_adapt(self, additional_in_channels):
        cpy = copy.deepcopy(self)

        l1 = self.first.dc.conv1
        l1_ = type(l1)( l1.in_channels + additional_in_channels, l1.out_channels, 
                        kernel_size=l1.kernel_size, 
                        padding=l1.padding, 
                        padding_mode=l1.padding_mode)
        l1_.weight.data[:, :-additional_in_channels] = l1.weight.data
        cpy.first.dc.conv1 = l1_
        
        l1 = self.first.conv
        l1_ = type(l1)( l1.in_channels + additional_in_channels, l1.out_channels, 
                        kernel_size=l1.kernel_size, 
                        padding=l1.padding, 
                        padding_mode=l1.padding_mode)
        l1_.weight.data[:, :-additional_in_channels] = l1.weight.data
        cpy.first.conv = l1_

        return cpy

