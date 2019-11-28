import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
import re


class ConvBlock3d(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_rate=None):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.bn1 = nn.BatchNorm3d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_ch, out_ch*4, kernel_size=1, stride=1, padding=0)

        self.bn2 = nn.BatchNorm3d(out_ch*4)
        self.conv2 = nn.Conv3d(out_ch*4, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)

        if self.dropout_rate:
            x = nn.Dropout3d(self.dropout_rate)(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        if self.dropout_rate:
            x = nn.Dropout3d(self.dropout_rate)(x)

        return x


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1)))


class DenseNet3D(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(3, 4, 12, 8),
                 num_init_features=96, bn_size=4, drop_rate=0, num_classes=3):

        super(DenseNet3D, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool3d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out


class DenseUnet3D(nn.Module):
    def __init__(self, in_ch=4, num_classes=3):
        super().__init__()
        num_init_features = 96
        backbone = DenseNet3D(num_init_features=num_init_features)
        self.first_convblock = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(in_ch, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            # ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        self.pool0 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.denseblock1 = backbone.features.denseblock1
        self.transition1 = backbone.features.transition1
        self.denseblock2 = backbone.features.denseblock2
        self.transition2 = backbone.features.transition2
        self.denseblock3 = backbone.features.denseblock3
        self.transition3 = backbone.features.transition3
        self.denseblock4 = backbone.features.denseblock4
        self.bn5 = backbone.features.norm5
        self.relu = nn.ReLU(inplace=True)

        self.convblock4 = ConvBlock3D(504, 504, kernel_size=3, stride=1, padding=1)
        self.convblock3 = ConvBlock3D(504, 224, kernel_size=3, stride=1, padding=1)
        self.convblock2 = ConvBlock3D(224, 192, kernel_size=3, stride=1, padding=1)
        self.convblock1 = ConvBlock3D(192, 96, kernel_size=3, stride=1, padding=1)
        self.convblock0 = ConvBlock3D(96, 64, kernel_size=3, stride=1, padding=1)
        self.final_conv = nn.Conv3d(64, num_classes, kernel_size=1, stride=1)

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

        x = nn.functional.interpolate(x, scale_factor=(2, 2, 1), mode='trilinear', align_corners=True)
        # db3 = self.conv3(db3)
        # db43 = torch.add(up4, db3)
        x = self.convblock4(x)

        x = nn.functional.interpolate(x, scale_factor=(2, 2, 1), mode='trilinear', align_corners=True)
        # db3 = self.conv3(db3)
        # db43 = torch.add(up4, db3)
        x = self.convblock3(x)

        x = nn.functional.interpolate(x, scale_factor=(2, 2, 1), mode='trilinear', align_corners=True)
        # db3 = self.conv3(db3)
        # db43 = torch.add(up4, db3)
        x = self.convblock2(x)

        x = nn.functional.interpolate(x, scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        # db3 = self.conv3(db3)
        # db43 = torch.add(up4, db3)
        x = self.convblock1(x)

        x = nn.functional.interpolate(x, scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        # db3 = self.conv3(db3)
        # db43 = torch.add(up4, db3)
        x = self.convblock0(x)

        out = self.final_conv(x)

        return x, out


class ConvBlock3D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn(x)
        x = self.relu(x)
        return x


if __name__ == '__main__':
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '4'

    # model_bb = torchvision.models.densenet161(pretrained=True).cuda()

    model = DenseUnet3D().cuda()
    data = torch.randn((2, 3, 224, 224, 12)).cuda()
    pred = model(data)
    # print(pred.shape)
