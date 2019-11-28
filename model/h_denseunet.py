import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import sys
sys.path.append('..')
from model.denseunet import DenseUnet
from model.denseunet3d import DenseUnet3D


def slice(x, h1, h2):
    """ Define a tensor slice function
    """
    return x[:, 0, :, :, h1:h2]


def slice2d(x, h1, h2):

    tmp = x[h1:h2,:,:,:]
    tmp = tmp.permute(1, 2, 3, 0)
    tmp = tmp.unsqueeze(dim=0)
    return tmp


def slice_last(x):
    x = x[:,:,:,:,0]
    return x


def volume2seq(volume):
    """
    transform 3D volume to 2D slice sequences for 2d-denseunet
    :param volume: [batch_size, channel, h, w, t] (only support batch_size=1)
    :return:
    """
    input2d = slice(volume, 0, 2)
    single = slice(volume, 0, 1)
    input2d = torch.cat([single, input2d], dim=3)
    for i in range(volume.shape[4] - 2):
        input2d_tmp = slice(volume, i, i+3)
        input2d = torch.cat([input2d, input2d_tmp], dim=0)
        if i == volume.shape[4] - 3:
            final1 = slice(volume, volume.shape[4] -2, volume.shape[4])
            final2 = slice(volume, volume.shape[4] -1, volume.shape[4])
            final = torch.cat([final1, final2], dim=3)
            input2d = torch.cat([input2d, final], dim=0)
    input2d = input2d.permute(0, 3, 1, 2)
    return input2d


def seq2volume(seq, t):
    """
    transform 2D slice sequences to 3D volume for 3d-denseunet
    :param seq: [batch_size*t, channel, h, w] (only support batch_size=1)
    :param t: thickness of the 3D volume
    :return: [1, channel, h, w, t]
    """
    input3d = slice2d(seq, 0, 1)
    for j in range(t - 1):
        input3d_tmp = slice2d(seq, j+1, j+2)
        input3d = torch.cat([input3d, input3d_tmp], dim=4)
    return input3d


class HDenseUnet(nn.Module):
    def __init__(self, num_classes=3):
        super(HDenseUnet, self).__init__()
        self.denseunet = DenseUnet(in_ch=3, num_classes=3)
        self.denseunet3d = DenseUnet3D(in_ch=4, num_classes=3)

        self.final_conv = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1)
        self.final_bn = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Conv3d(64, num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        input2d = volume2seq(x)
        feature2d, classifier2d = self.denseunet(input2d)
        # print('feature2d.shape ori', feature2d.shape)
        # print('classifier2d.shape ori', classifier2d.shape)
        feature2d = seq2volume(feature2d, x.shape[4])
        classifier2d = seq2volume(classifier2d, x.shape[4])
        # print('feature2d.shape trans', feature2d.shape)
        # print('classifier2d.shape trans', classifier2d.shape)

        input3d = torch.cat([x, classifier2d], dim=1)
        # print('input3d', input3d.shape)
        feature3d, classifier3d = self.denseunet3d(input3d)
        # print('feature3d.shape ori', feature3d.shape)
        # print('classifier3d.shape ori', classifier3d.shape)

        final = torch.add(feature3d, feature2d)
        final = self.final_conv(final)
        final = nn.Dropout3d(p=0.3)(final)
        final = self.final_bn(final)
        final = self.relu(final)
        final = self.classifier(final)
        # print(final.shape)

        return final


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'

    x = torch.randn((1, 1, 224, 224, 12)).cuda()
    # input2d = volume2seq(x)
    # print(input2d.shape)
    # feature2d = seq2volume(input2d, x.shape[4])
    # print(feature2d.shape)
    model = HDenseUnet(num_classes=3).cuda()
    for i in range(100):
        y = model(x)
        print(y.shape)



