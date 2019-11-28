from model.components import *

from torchvision.models import densenet121, resnet18, resnet34, resnet50, resnet101, vgg11, vgg16

def unet11(pre_train=True, bn=False, se=False, residual=False, **kwargs):
    """
    pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with VGG11
            carvana - all weights are pre-trained on
                Kaggle: Carvana dataset https://www.kaggle.com/c/carvana-image-masking-challenge
    """
    vgg_model = vgg11(pretrained=pre_train)

    if bn:
        return UNet11BN(backbone=vgg_model, **kwargs)

    return UNet11(backbone=vgg_model, **kwargs)


class UNet11:
    def __init__(self, backbone, n_classes=1, num_filters=32):
        """
        :param backbone:
        :param n_classes:
        :param num_filters:
        """
        super().__init__()
        if n_classes == 2:
            n_classes = 1

        self.n_classes = n_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = backbone.features
        self.relu = self.encoder[1]

        self.inc = nn.Sequential(
            self.encoder[0],   # Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu)
        self.down1 = nn.Sequential(
            self.pool,
            self.encoder[3],  # Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu)
        self.down2 = nn.Sequential(
            self.pool,
            self.encoder[6],  # Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu,
            self.encoder[8],  # Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu)
        self.down3 = nn.Sequential(
            self.pool,
            self.encoder[11],  # Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu,
            self.encoder[13],  # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu)
        self.down4 = nn.Sequential(
            self.pool,
            self.encoder[16],  # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu,
            self.encoder[18],  # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu)
        self.down5 = nn.Sequential(
            self.pool,   # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ConvRelu(num_filters * 16, num_filters * 16))

        self.up5 = TernausUp(num_filters * 16, num_filters * 8, num_filters * 16, num_filters * 16)
        self.up4 = TernausUp(num_filters * 16, num_filters * 8, num_filters * 16, num_filters * 16)
        self.up3 = TernausUp(num_filters * 16, num_filters * 4, num_filters * 8, num_filters * 8)
        self.up2 = TernausUp(num_filters * 8, num_filters * 2, num_filters * 4, num_filters * 4)
        self.up1 = TernausUp(num_filters * 4, num_filters * 1, num_filters * 2, num_filters)
        self.outc = nn.Conv2d(num_filters, n_classes, kernel_size=1)
        self.up_features = [128, 256, 512, 512, 512]


class UNet11BN:
    """
    this class only add bn layer after each encoder of unet11
    it should be merged to unet11 class
    """
    def __init__(self, backbone, n_classes=1, num_filters=32):
        """
        :param backbone:
        :param n_classes:
        :param num_filters:
        """
        super().__init__()
        if n_classes == 2:
            n_classes = 1
        self.n_classes = n_classes
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = backbone.features

        self.relu = self.encoder[1]
        self.inc = nn.Sequential(  # Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.encoder[0],
            nn.BatchNorm2d(64),
            self.relu)  # relu
        self.down1 = nn.Sequential(self.pool,
                                   # Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                   self.encoder[3],
                                   nn.BatchNorm2d(128),
                                   self.relu)
        self.down2 = nn.Sequential(self.pool,
                                   # Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                   self.encoder[6],
                                   nn.BatchNorm2d(256),
                                   self.relu,
                                   # Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                   self.encoder[8],
                                   nn.BatchNorm2d(256),
                                   self.relu)
        self.down3 = nn.Sequential(self.pool,
                                   # Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                   self.encoder[11],
                                   nn.BatchNorm2d(512),
                                   self.relu,
                                   # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                   self.encoder[13],
                                   nn.BatchNorm2d(512),
                                   self.relu)
        self.down4 = nn.Sequential(self.pool,
                                   # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                   self.encoder[16],
                                   nn.BatchNorm2d(512),
                                   self.relu,
                                   # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                   self.encoder[18],
                                   nn.BatchNorm2d(512),
                                   self.relu)
        self.down5 = nn.Sequential(self.pool,
                                   # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
                                   ConvRelu(num_filters * 16, num_filters * 16))

        self.up5 = TernausUp(num_filters * 16, num_filters * 8, num_filters * 16, num_filters * 16, bn=True)
        self.up4 = TernausUp(num_filters * 16, num_filters * 8, num_filters * 16, num_filters * 16, bn=True)
        self.up3 = TernausUp(num_filters * 16, num_filters * 4, num_filters * 8, num_filters * 8, bn=True)
        self.up2 = TernausUp(num_filters * 8, num_filters * 2, num_filters * 4, num_filters * 4, bn=True)
        self.up1 = TernausUp(num_filters * 4, num_filters * 1, num_filters * 2, num_filters, bn=True)
        self.outc = nn.Conv2d(num_filters, n_classes, kernel_size=1)
        self.up_features = [128, 256, 512, 512, 512]


class UNet16(nn.Module):
    def __init__(self, backbone, n_classes=1, num_filters=32, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG16
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        if n_classes == 2:
            n_classes = 1

        self.n_classes = n_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder = backbone.features
        self.relu = nn.ReLU(inplace=True)

        self.inc = nn.Sequential(
            self.encoder[0],  # Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu,
            self.encoder[2],  # Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu
        )
        self.down1 = nn.Sequential(
            self.pool,
            self.encoder[5],  # Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu,
            self.encoder[7],  # Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu
        )
        self.down2 = nn.Sequential(
            self.pool,
            self.encoder[10],  # Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu,
            self.encoder[12],  # Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu,
            self.encoder[14],  # Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu
        )
        self.down3 = nn.Sequential(
            self.pool,
            self.encoder[17],  # Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu,
            self.encoder[19],  # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu,
            self.encoder[21],  # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu
        )
        self.down4 = nn.Sequential(
            self.pool,
            self.encoder[24],  # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu,
            self.encoder[26],  # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu,
            self.encoder[28],  # Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            self.relu
        )

        self.down5 = nn.Sequential(
            self.pool,
            ConvRelu(512, num_filters * 16)
        )

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, n_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec1), dim=1)
        else:
            x_out = self.final(dec1)

        return x_out


class AlbuNet(nn.Module):
    """
        UNet (https://arxiv.org/abs/1505.04597) with Resnet34(https://arxiv.org/abs/1512.03385) encoder
        Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/
        """

    def __init__(self, num_classes=1, num_filters=32, pretrained=False, is_deconv=False):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with resnet34
        :is_deconv:
            False: bilinear interpolation is used in decoder
            True: deconvolution is used in decoder
        """
        super().__init__()
        self.num_classes = num_classes

        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = resnet34(pretrained=pretrained)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu,
                                   self.pool)

        self.conv2 = self.encoder.layer1

        self.conv3 = self.encoder.layer2

        self.conv4 = self.encoder.layer3

        self.conv5 = self.encoder.layer4

        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)

        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(256 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(128 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(64 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2, is_deconv)
        self.dec1 = DecoderBlockV2(num_filters * 2 * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec0 = ConvRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))

        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(self.final(dec0), dim=1)
        else:
            x_out = self.final(dec0)

        return x_out

