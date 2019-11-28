import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import torchvision

warnings.filterwarnings(action='ignore')


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch, reduction=16):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.se = SELayer(out_ch, reduction)

        self.channel_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = self.se(x)

        if residual.shape[1] != x.shape[1]:
            residual = self.channel_conv(residual)
        x += residual
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=False, rate=0.1):
        super(outconv, self).__init__()
        self.dropout = dropout
        if dropout:
            print('dropout', rate)
            self.dp = nn.Dropout2d(rate)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        if self.dropout:
            x = self.dp(x)
        x = self.conv(x)
        return x


class UNetRes101(nn.Module):
    def __init__(self, n_channels, n_classes, deep_supervision=False, dropout=False, rate=0.1, bn=False, pretrain=True):
        super(UNetRes101, self).__init__()
        self.deep_supervision = deep_supervision
        self.vgg_features = torchvision.models.resnet50(pretrained=pretrain).features
        # print(self.vgg_features)
        self.inc = self.vgg_features[:4]
        self.down1 = self.vgg_features[4:9]
        self.down2 = self.vgg_features[9:16]
        self.down3 = self.vgg_features[16:23]
        self.down4 = self.vgg_features[23:30]
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes, dropout, rate)
        self.dsoutc4 = outconv(256, n_classes)
        self.dsoutc3 = outconv(128, n_classes)
        self.dsoutc2 = outconv(64, n_classes)
        self.dsoutc1 = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x44 = self.up1(x5, x4)
        x33 = self.up2(x44, x3)
        x22 = self.up3(x33, x2)
        x11 = self.up4(x22, x1)
        x0 = self.outc(x11)
        if self.deep_supervision and self.training:
            x11 = F.interpolate(self.dsoutc1(x11), x0.shape[2:], mode='bilinear')
            x22 = F.interpolate(self.dsoutc2(x22), x0.shape[2:], mode='bilinear')
            x33 = F.interpolate(self.dsoutc3(x33), x0.shape[2:], mode='bilinear')
            x44 = F.interpolate(self.dsoutc4(x44), x0.shape[2:], mode='bilinear')

            return x0, x11, x22, x33, x44
        else:
            return x0


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'


    def weights_init(m):
        classname = m.__class__.__name__
        # print(classname)
        if classname.find('Conv') != -1:
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)


    model = VGGUNet(3, 3, deep_supervision=True).cuda()
    model.apply(weights_init)

    x = torch.randn((1, 3, 512, 512)).cuda()

    for i in range(1000):
        y0, y1, y2, y3, y4 = model(x)
        print(y0.shape, y1.shape, y2.shape, y3.shape, y4.shape)

# print(model.weight)

# for name, param in model.named_parameters():
#     print(name, param.requires_grad)

# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# loss_func = torch.nn.CrossEntropyLoss()
#
# # model.apply(weights_init)
# # # model.load_state_dict(torch.load('./MODEL.pth'))
# model = model.cuda()
# print('load done')
# input()
#
# model.train()
# for i in range(100):
# 	x = torch.randn(1, 3, 256, 256).cuda()
# 	label = torch.rand(1, 256, 256).long().cuda()
# 	y = model(x)
# 	print(i)
#
# 	loss = loss_func(y, label)
# 	optimizer.zero_grad()
# 	loss.backward()
# 	optimizer.step()
#
# print('train done')
# input()
#
# with torch.no_grad():
# 	model.eval()
# 	for i in range(1000):
# 		x = torch.randn(1, 1, 256, 256).cuda()
# 		label = torch.rand(1, 256, 256).long().cuda()
# 		y = model(x)
# 		print(y.shape)
#
# 		# loss = loss_func(y, label)
# 		# optimizer.zero_grad()
# 		# loss.backward()
# 		# optimizer.step()

# input()

