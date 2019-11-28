import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

warnings.filterwarnings(action='ignore')


class double_conv(nn.Module):
    # conv -> bn -> relu * 2

    def __init__(self, in_ch, out_ch, ks=3):
        super(double_conv, self).__init__()
        padding = (ks - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=ks, padding=padding),
            nn.BatchNorm3d(out_ch),
            nn.PReLU(out_ch),
            nn.Conv3d(out_ch, out_ch, kernel_size=ks, padding=padding),
            nn.BatchNorm3d(out_ch),
            nn.PReLU(out_ch),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class bottle_neck(nn.Module):
    # 1*1 -> 3*3 -> 1*1
    def __init__(self, in_chan, out_chan, ks=3):
        super(bottle_neck, self).__init__()
        padding = (ks - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv3d(in_chan, out_chan // 2, kernel_size=1),
            nn.BatchNorm3d(out_chan // 2),
            nn.PReLU(out_chan // 2),

            nn.Conv3d(out_chan // 2, out_chan // 2, kernel_size=3, padding=padding),
            nn.BatchNorm3d(out_chan // 2),
            nn.PReLU(out_chan // 2),

            nn.Conv3d(out_chan // 2, out_chan, kernel_size=1),
            nn.BatchNorm3d(out_chan),
            nn.PReLU(out_chan),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)
        self.channel_conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x
        x = self.conv(x)
        if identity.shape != x.shape:
            identity = self.channel_conv(identity)
        return x + identity


class down(nn.Module):
    def __init__(self, in_ch, out_ch, double=True):
        super(down, self).__init__()
        self.down_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 2, 2),
            nn.BatchNorm3d(out_ch),
            nn.PReLU(out_ch),
        )
        if double:
            self.conv = double_conv(out_ch, out_ch)
        else:
            self.conv = bottle_neck(out_ch, out_ch)

        self.channel_conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):

        x = self.down_conv(x)
        identity = x
        x = self.conv(x)

        if identity.shape != x.shape:
            identity = self.channel_conv(identity)

        return x + identity


class up(nn.Module):
    def __init__(self, in_ch, out_ch, double=True):
        super(up, self).__init__()

        self.up = nn.ConvTranspose3d(in_ch, out_ch // 2, 2, stride=2)

        if double:
            self.conv = double_conv(out_ch, out_ch)
        else:
            self.conv = bottle_neck(out_ch, out_ch)

        self.channel_conv = nn.Conv3d(out_ch // 2, out_ch, 1)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffZ // 2, diffZ - diffZ // 2,
                        diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2, ])

        identity = x1

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        if identity.shape != x.shape:
            identity = self.channel_conv(identity)

        return x + identity


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class Vnet(nn.Module):
    def __init__(self, n_channels, n_classes, double=True):
        super(Vnet, self).__init__()
        self.inc = inconv(n_channels, 16)
        self.down1 = down(16, 32, double)
        self.down2 = down(32, 64, double)
        self.down3 = down(64, 128, double)
        self.down4 = down(128, 256, double)

        self.up1 = up(256, 256, double)
        self.up2 = up(256, 128, double)
        self.up3 = up(128, 64, double)
        self.up4 = up(64, 32, double)

        self.outconv = outconv(32, n_classes)

    def forward(self, x):
        # print(x.shape)
        x1 = self.inc(x)
        # print(x1.shape)
        x2 = self.down1(x1)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        x5 = self.down4(x4)
        # print(x5.shape)

        x44 = self.up1(x5, x4)
        # print(x44.shape)
        x33 = self.up2(x44, x3)
        # print(x33.shape)
        x22 = self.up3(x33, x2)
        # print(x22.shape)
        x11 = self.up4(x22, x1)
        # print(x11.shape)

        x0 = self.outconv(x11)
        # print(x0.shape)
        return x0


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'


    def weights_init(m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Conv') != -1:
            torch.nn.init.kaiming_normal(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)


    model = Vnet(3, 3, double=False).cuda()
    model.apply(weights_init)

    x = torch.randn((1, 3, 128, 128, 64)).cuda()

    for i in range(1):
        y0 = model(x)
        print(y0.shape)






