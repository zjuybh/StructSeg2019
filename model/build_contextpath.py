import torch
from torchvision import models
from torchvision.models.resnet import BasicBlock


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
        # # # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature2, feature3, feature4, tail


class resnet34(torch.nn.Module):
    def __init__(self, in_channel=3, pretrained=True):
        super().__init__()
        self.features = models.resnet34(pretrained=pretrained)
        self.conv1 = torch.nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                                     bias=False)
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)             # 1 / 4
        feature2 = self.layer2(feature1)      # 1 / 8
        feature3 = self.layer3(feature2)      # 1 / 16
        feature4 = self.layer4(feature3)      # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature2, feature3, feature4, tail


class resnet50(torch.nn.Module):
    def __init__(self, in_channel=3, pretrained=True):
        super().__init__()
        self.features = models.resnet50(pretrained=pretrained)
        self.conv1 = torch.nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                                     bias=False)
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)             # 1 / 4
        feature2 = self.layer2(feature1)      # 1 / 8
        feature3 = self.layer3(feature2)      # 1 / 16
        feature4 = self.layer4(feature3)      # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature2, feature3, feature4, tail


class resnet101(torch.nn.Module):
    def __init__(self, in_channel=3, pretrained=True):
        super().__init__()
        self.features = models.resnet101(pretrained=pretrained)
        self.conv1 = torch.nn.Conv2d(in_channel, 64, kernel_size=7, stride=2, padding=3,
                                     bias=False)
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        feature1 = self.layer1(x)             # 1 / 4
        feature2 = self.layer2(feature1)      # 1 / 8
        feature3 = self.layer3(feature2)      # 1 / 16
        feature4 = self.layer4(feature3)      # 1 / 32
        # global average pooling to build tail
        # tail = torch.mean(feature4, 3, keepdim=True)
        # tail = torch.mean(tail, 2, keepdim=True)
        tail = self.avgpool(feature4)
        return feature2, feature3, feature4, tail


def build_contextpath(name, in_channel=3):
    if name == 'resnet18':
        return resnet18(in_channel=in_channel, pretrained=True)
    elif name == 'resnet34':
        return resnet34(in_channel=in_channel, pretrained=True)
    elif name == 'resnet50':
        return resnet50(in_channel=in_channel, pretrained=True)
    elif name == 'resnet101':
        return resnet101(in_channel=in_channel, pretrained=True)
    else:
        raise Exception('not implemented for', name)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'

    model = build_contextpath('xception').cuda()
    x = torch.rand(2, 3, 512, 512).cuda()
    cx1, cx2, cx3, tail = model(x)
    print(cx1.shape)
    print(cx2.shape)
    print(cx3.shape)
    print(tail.shape)
