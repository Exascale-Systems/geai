import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Helper Functions ---
def conv2d(in_c, out_c, k=3, s=1, p=1):
    return nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False)


def down2d(in_c, out_c):
    return nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False)


def deconv3d(in_c, out_c):
    return nn.ConvTranspose3d(
        in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False
    )


# --- Building Blocks ---
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
    def __init__(self, in_channels=1):
        super().__init__()
        ch = [128, 256, 512, 1024]
        self.first = down2d(in_channels, ch[0])
        stages = []
        in_c = ch[0]
        for out_c in ch[1:]:
            stages += [ResBlock2D(in_c), down2d(in_c, out_c)]
            in_c = out_c
        self.stages = nn.Sequential(*stages)
        self.out_channels = ch[-1]

    def forward(self, x):
        x = self.first(x)
        x = self.stages(x)
        return x


class DimTransform(nn.Module):
    def __init__(self, c=1024):
        super().__init__()
        self.conv2d_1x1 = nn.Conv2d(c, c, kernel_size=1)
        self.conv3d_1x1 = nn.Conv3d(c, c, kernel_size=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, f2):
        f2 = self.act(self.conv2d_1x1(f2))
        f3 = f2.unsqueeze(2)
        f3 = self.act(self.conv3d_1x1(f3))
        return f3


class Decoder3D(nn.Module):
    def __init__(self, in_c=1024):
        super().__init__()
        ch = [512, 256, 128, 64]
        ups = []
        c = in_c
        for out_c in ch:
            ups += [
                deconv3d(c, out_c),
                nn.BatchNorm3d(out_c),
                nn.LeakyReLU(0.1, inplace=True),
            ]
            c = out_c
        self.ups = nn.Sequential(*ups)
        self.head = nn.Conv3d(c, 1, kernel_size=1)

    def forward(self, x):
        x = self.ups(x)
        x = self.head(x)
        return x.squeeze(1)


# --- Main Network ---
class GravInvNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.enc = Encoder2D(in_channels=in_channels)
        self.dim = DimTransform(self.enc.out_channels)
        self.dec = Decoder3D(self.enc.out_channels)

    def forward(self, gz):
        """
        gz:   (B,C,32,32)
        pred: (B,16,32,32)
        """
        f2 = self.enc(gz)
        assert f2.shape[-2:] == (2, 2), f"Encoder produced {f2.shape}"
        f3 = self.dim(f2)
        pred = self.dec(f3)
        return pred
