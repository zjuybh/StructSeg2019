import torch
from torch import nn
import sys
sys.path.append('..')
from model.build_contextpath import build_contextpath
import warnings
import torch.nn.functional as F
import numpy as np
import time
from torchvision import models
warnings.filterwarnings(action='ignore')


# class ConvBlock(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, reduction=16):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU()
#
#     def forward(self, input):
#         x = self.conv1(input)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x
#
#
# class AttentionRefinementModule(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.sigmoid = nn.Sigmoid()
#         self.in_channels = in_channels
#         self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#         self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
#
#     def forward(self, input):
#         # global average pooling
#         x = self.avgpool(input)
#         assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
#         x = self.conv(x)
#         # x = self.sigmoid(self.bn(x))
#         x = self.sigmoid(x)
#         # channels of input and x should be same
#         x = torch.mul(input, x)
#
#         # spatial attention 1
#         mean_feature = torch.mean(x, dim=1, keepdim=True)
#         max_feature, _ = torch.max(x, dim=1, keepdim=True)
#         spatial_feature = torch.cat((mean_feature, max_feature), dim=1)
#         spatial_attention = self.spatial_conv(spatial_feature)
#         spatial_attention = self.sigmoid(spatial_attention)
#         x = torch.mul(x, spatial_attention)
#
#         return x
#
#
# class FeatureFusionModule(torch.nn.Module):
#     def __init__(self, num_classes, context_path):
#         super().__init__()
#         # self.in_channels = input_1.channels + input_2.channels
#         if context_path in ['resnet18', 'resnet34']:
#             self.in_channels = 896  # (128) + 256(448) + 256 + 512
#         elif context_path in ['xception']:
#             self.in_channels = 2008
#         else:
#             self.in_channels = 3584  # (512) + 256 + 1024 + 2048
#         self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1, reduction=1)
#         self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()
#         self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
#
#     def forward(self, input_0, input_1, input_2):
#         x = torch.cat((input_0, input_1, input_2), dim=1)
#         assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
#         feature = self.convblock(x)
#         x = self.avgpool(feature)
#
#         x = self.relu(self.conv1(x))
#         x = self.sigmoid(self.conv2(x))
#         x = torch.mul(feature, x)
#
#         x = torch.add(x, feature)
#
#         return x


class resnet18(torch.nn.Module):
    def __init__(self, in_channel=3, pretrained=True):
        super().__init__()
        self.features = models.resnet18(pretrained=pretrained)
        # self.conv1 = self.features.conv1
        self.conv1 = torch.nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4
        # self.layer5 = self.features._make_layer(BasicBlock, 1024, 2, stride=2)

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)             # 1 / 4
        feature2 = self.layer2(feature1)      # 1 / 8
        feature3 = self.layer3(feature2)      # 1 / 16
        feature4 = self.layer4(feature3)      # 1 / 32
        return feature4


class Res18(torch.nn.Module):
    def __init__(self, num_classes, in_channel=3):
        super().__init__()
        # build context path
        self.context_path = resnet18(in_channel=in_channel)

        # # build attention refinement module
        # if context_path in ['resnet18', 'resnet34']:
        #     self.attention_refinement_module0 = AttentionRefinementModule(128, 128)
        #     self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
        #     self.attention_refinement_module2 = AttentionRefinementModule(512, 512)
        # elif context_path in ['xception']:
        #     self.attention_refinement_module0 = AttentionRefinementModule(256, 256)
        #     self.attention_refinement_module1 = AttentionRefinementModule(728, 728)
        #     self.attention_refinement_module2 = AttentionRefinementModule(1024, 1024)
        # else:
        #     self.attention_refinement_module0 = AttentionRefinementModule(512, 512)
        #     self.attention_refinement_module1 = AttentionRefinementModule(1024, 1024)
        #     self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048)
        #
        #
        # # build feature fusion module
        # self.feature_fusion_module = FeatureFusionModule(num_classes, context_path)

        # build final convolution
        self.conv = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)

        # # supervision block
        # if context_path in ['resnet18', 'resnet34']:
        #     self.supervision1 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
        #     self.supervision2 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
        # elif context_path in ['xception']:
        #     self.supervision1 = nn.Conv2d(in_channels=728, out_channels=num_classes, kernel_size=1)
        #     self.supervision2 = nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1)
        # else:
        #     self.supervision1 = nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1)
        #     self.supervision2 = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)

    def forward(self, input):
        # output of context path
        cx = self.context_path(input)
        # cx0 = self.attention_refinement_module0(cx0)
        # cx1 = self.attention_refinement_module1(cx1)
        # cx2 = self.attention_refinement_module2(cx2)
        # cx2 = torch.mul(cx2, tail)
        #
        # # upsampling
        # cx1 = torch.nn.functional.interpolate(cx1, scale_factor=2, mode='bilinear')
        # cx2 = torch.nn.functional.interpolate(cx2, scale_factor=4, mode='bilinear')
        #
        # if self.training:
        #     cx1_sup = self.supervision1(cx1)
        #     cx2_sup = self.supervision2(cx2)
        #     cx1_sup = torch.nn.functional.interpolate(cx1_sup, scale_factor=8, mode='bilinear')
        #     cx2_sup = torch.nn.functional.interpolate(cx2_sup, scale_factor=8, mode='bilinear')
        #
        # # output of feature fusion module
        # result = self.feature_fusion_module(cx0, cx1, cx2)
        #
        # # upsampling
        result = self.conv(cx)
        result = torch.nn.functional.interpolate(cx, scale_factor=32, mode='bilinear')

        if self.training:
            return result

        return result


import torch
from torch import nn
from model.build_contextpath import build_contextpath
import warnings
import torch.nn.functional as F
warnings.filterwarnings(action='ignore')


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn(x)
        x = self.relu(x)
        return x


class BiSeNet(torch.nn.Module):
    def __init__(self, num_classes, in_channel=3):
        super().__init__()
        self.in_channels = 512
        # build context path
        self.context_path = resnet18(in_channel=in_channel)

        self.convblock = ConvBlock(in_channels=self.in_channels, out_channels=num_classes, stride=1, reduction=1)

        # build final convolution
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)

    def forward(self, input):
        # output of context path
        cx2 = self.context_path(input)
        cx2 = torch.nn.functional.interpolate(cx2, scale_factor=4, mode='bilinear')

        # output of feature fusion module
        result = self.convblock(cx2)

        # upsampling
        result = torch.nn.functional.interpolate(result, scale_factor=8, mode='bilinear')
        result = self.conv(result)

        return result


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    # model = BiSeNet(1, 'xception')
    model = SSN(1, 'resnet18')
    # model = nn.DataParallel(model)

    model = model.cuda()
    # model.train()
    model.eval()
    # for key in model.named_parameters():
    #     print(key[1].device)
    time_list = []
    for i in range(1000):
        x = torch.rand(1, 3, 512, 512).cuda()
        # x = torch.rand(1, 3, 640, 352).cuda()
        # edge = torch.rand(2, 1, 512, 512).cuda()

        # record = model.parameters()
        # for key, params in model.named_parameters():
        #     if 'bn' in key:
        #         params.requires_grad = False
        # y1, y2, y3 = model(x)
        tic = time.time()
        y = model(x)
        toc = time.time()
        time_list.append(toc-tic)
        print(y.shape)
        input()
        # print(y1.shape)
        # print(y2.shape)
        # print(y3.shape)

    print('avg time', np.mean(time_list))
