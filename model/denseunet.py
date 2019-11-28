import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict


class DenseUnet(nn.Module):
    def __init__(self, in_ch=3, num_classes=3, hybrid=False):
        super().__init__()
        self.hybrid = hybrid
        num_init_features = 96
        backbone = torchvision.models.densenet161(pretrained=True)
        self.first_convblock = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(in_ch, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            # ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.denseblock1 = backbone.features.denseblock1
        self.transition1 = backbone.features.transition1
        self.denseblock2 = backbone.features.denseblock2
        self.transition2 = backbone.features.transition2
        self.denseblock3 = backbone.features.denseblock3
        self.transition3 = backbone.features.transition3
        self.denseblock4 = backbone.features.denseblock4
        self.bn5 = backbone.features.norm5
        self.relu = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(2112, 2208, kernel_size=1, stride=1)

        self.convblock43 = ConvBlock(2208, 768)
        self.convblock32 = ConvBlock(768, 384, kernel_size=3, stride=1, padding=1)
        self.convblock21 = ConvBlock(384, 96, kernel_size=3, stride=1, padding=1)
        self.convblock10 = ConvBlock(96, 96, kernel_size=3, stride=1, padding=1)
        self.convblock00 = ConvBlock(96, 64, kernel_size=3, stride=1, padding=1)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1)

    def forward(self, x):
        db0 = self.first_convblock(x)
        x = self.pool0(db0)
        db1 = self.denseblock1(x)
        x = self.transition1(db1)
        db2 = self.denseblock2(x)
        x = self.transition2(db2)
        db3 = self.denseblock3(x)
        x = self.transition3(db3)
        x = self.denseblock4(x)
        x = self.bn5(x)
        db4 = self.relu(x)

        up4 = torch.nn.functional.interpolate(db4, scale_factor=2, mode='bilinear', align_corners=True)
        db3 = self.conv3(db3)
        db43 = torch.add(up4, db3)
        db43 = self.convblock43(db43)

        up43 = torch.nn.functional.interpolate(db43, scale_factor=2, mode='bilinear', align_corners=True)
        db432 = torch.add(up43, db2)
        db432 = self.convblock32(db432)

        up432 = torch.nn.functional.interpolate(db432, scale_factor=2, mode='bilinear', align_corners=True)
        db4321 = torch.add(up432, db1)
        db4321 = self.convblock21(db4321)

        up4321 = torch.nn.functional.interpolate(db4321, scale_factor=2, mode='bilinear', align_corners=True)
        db43210 = torch.add(up4321, db0)
        db43210 = self.convblock10(db43210)

        up43210 = torch.nn.functional.interpolate(db43210, scale_factor=2, mode='bilinear', align_corners=True)
        db43210 = self.convblock00(up43210)

        out = self.final_conv(db43210)

        if self.hybrid:
            return db43210, out
        else:
            return out


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn(x)
        x = self.relu(x)
        return x



class Dense_Block(nn.Module):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
            # Arguments
                x: input tensor
                stage: index for dense block
                nb_layers: the number of layers of conv_block to append to the model.
                nb_filter: number of filters
                growth_rate: growth rate
                dropout_rate: dropout rate
                weight_decay: weight decay factor
                grow_nb_filters: flag to decide to allow number of filters to grow
    '''
    def __init__(self, x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1e-4,
                 grow_nb_filters=True):
        super().__init__()
        torchvision.models.densenet161()

    def forward(self, x):
        pass


if __name__ == '__main__':
    # model = torchvision.models.densenet161()
    # for name, param in model.named_parameters():
    #     print(name)

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    # model_bb = torchvision.models.densenet161(pretrained=True).cuda()

    model = DenseUnet().cuda()
    data = torch.randn((2, 3, 224, 224)).cuda()
    pred = model(data)
    print(pred.shape)


