import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        # assume input is (224, 224)
        self.down1 = (Down(64, 128))
        # (112, 112)
        self.down2 = (Down(128, 256))
        # (56, 56)
        self.down3 = (Down(256, 512))
        # (28, 28)
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        # (14, 14)
        self.up1 = (Up(1024, 512 // factor, bilinear))
        # (28, 28)
        self.up2 = (Up(512, 256 // factor, bilinear))
        # (56, 56)
        self.up3 = (Up(256, 128 // factor, bilinear))
        # (112, 112)
        self.up4 = (Up(128, 64, bilinear))
        # (224, 224)
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        output = self.outc(x9)
        return x1, x2, x3, x4, output

class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        self.vgg19 = models.vgg19(pretrained = True)
        num_features = self.vgg19.classifier[6].in_features
        self.vgg19.classifier[6] = nn.Linear(num_features, num_classes)
    def forward(self, x, y1, y2, y3, y4):
        x = self.vgg19.features[0:4](x)
        x = x + y1
        x = self.vgg19.features[4:9](x)
        x = x + y2
        x = self.vgg19.features[9:18](x)
        x = x + y3
        x = self.vgg19.features[18:27](x)
        x = x + y4
        x = self.vgg19.features[27:](x)
        x = self.vgg19.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.vgg19.classifier(x)
        return x


class GlauNet(nn.Module):
    def __init__(self):
        super(GlauNet, self).__init__()
        self.UNet = UNet(3, 3, True)
        self.VGG19 = VGG19(2)
        self.is_train = False
    def forward(self, x):
        y1, y2, y3, y4, o1 = self.UNet(x)
        if self.is_train == True:
            idx = random.randint(0, x.size()[0] - 1)
            pil_image = transforms.ToPILImage()(x[idx])
            path = f"union_GlauNet/img/input.jpg"
            pil_image.save(path)
            pil_image = transforms.ToPILImage()(o1[idx])
            path = f"union_GlauNet/img/output.jpg"
            pil_image.save(path)
        o2 = self.VGG19(x, y1, y2, y3, y4)
        return o1, o2

