import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d(in_c, out_c, k=3, s=1, p=1):
    return nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)

def down2d(in_c, out_c):  # 4x4, stride 2 (no output_padding)
    return nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)

class ResBlock2D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c)
        self.conv1 = conv2d(c, c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(c)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        y = self.act(self.bn1(x))
        y = self.conv1(y)
        y = self.act(self.bn2(y))
        return x + y

class Encoder2D(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [128, 256, 512, 1024, 2048]  # doubles each stage
        self.first = down2d(1, ch[0])  # 1x64x64 -> 128x32x32
        stages = []
        in_c = ch[0]
        for out_c in ch[1:]:
            stages += [ResBlock2D(in_c), down2d(in_c, out_c)]
            in_c = out_c
        self.stages = nn.Sequential(*stages)  # ends at 2048x2x2
        self.out_channels = ch[-1]
    def forward(self, x):
        x = self.first(x)
        x = self.stages(x)
        return x  # (B, 2048, 2, 2)

class DimTransform(nn.Module):
    def __init__(self, c=2048):
        super().__init__()
        self.conv2d_1x1 = nn.Conv2d(c, c, kernel_size=1)
        self.conv3d_1x1 = nn.Conv3d(c, c, kernel_size=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, f2):
        # f2: (B, C, 2, 2)
        f2 = self.act(self.conv2d_1x1(f2))
        f3 = f2.unsqueeze(2)  # (B, C, 1, 2, 2)
        f3 = self.act(self.conv3d_1x1(f3))
        return f3  # (B, C, 1, 2, 2)

def deconv3d(in_c, out_c):  # 4x4x4, stride 2
    return nn.ConvTranspose3d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)

class Decoder3D(nn.Module):
    def __init__(self, in_c=2048):
        super().__init__()
        ch = [1024, 512, 256, 128, 64]  # reduce channels as we upsample
        ups = []
        c = in_c
        for out_c in ch:
            ups += [deconv3d(c, out_c), nn.BatchNorm3d(out_c), nn.LeakyReLU(0.1, inplace=True)]
            c = out_c
        self.ups = nn.Sequential(*ups)  # up to spatial (32,64,64)
        self.head = nn.Conv3d(c, 1, kernel_size=1)  # 1 channel over depth
    def forward(self, x):
        # x: (B, 2048, 1, 2, 2) -> ... -> (B, 64, 32, 64, 64)
        x = self.ups(x)
        x = self.head(x)  # (B,1,32,64,64)
        return x.squeeze(1)  # (B, 32, 64, 64)

class GravInvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder2D()
        self.dim = DimTransform(self.enc.out_channels)
        self.dec = Decoder3D(self.enc.out_channels)
    def forward(self, gz):
        """
        gz: (B,1,64,64) -> pred: (B,32,64,64)
        """
        f2 = self.enc(gz)
        assert f2.shape[-2:] == (2,2), f"Encoder produced {f2.shape}"
        f3 = self.dim(f2)
        pred = self.dec(f3)
        return pred
