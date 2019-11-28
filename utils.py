import SimpleITK as sitk
import numpy as np
import cv2
import torch.nn as nn
import torch
from torch.nn import functional as F
from PIL import Image
import pandas as pd
import os


def setDicomWinWidthWinCenter(img_data, winwidth, wincenter):
    img_temp = img_data
    min = (2 * wincenter - winwidth) / 2.0 + 0.5
    max = (2 * wincenter + winwidth) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)

    img_temp = (img_temp - min) * dFactor

    min_index = img_temp < 0
    img_temp[min_index] = 0
    max_index = img_temp > 255
    img_temp[max_index] = 255

    return img_temp


def readDcmSeries(path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image)  # z, y, x
    image_array = setDicomWinWidthWinCenter(image_array, 300, -20)
    return image_array


def fill_edge(img):
    img = np.uint8(img)
    img_ori = img.copy()
    img_ori[img == 1] = 80
    img_ori[img == 2] = 160
    img_ori[img == 5] = 255

    img_1 = img.copy()
    img_1[img == 1] = 255
    img_1[img != 1] = 0
    mask = np.zeros([514, 514], np.uint8)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_GRAY2BGR)
    cv2.floodFill(img_1, mask, (0, 0), (100, 100, 100), (1, 1, 1), (101, 101, 101), cv2.FLOODFILL_FIXED_RANGE)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_1[img_1 == 0] = 255
    img_1[img_1 != 255] = 0

    img_2 = img.copy()
    img_2[img == 2] = 255
    img_2[img != 2] = 0
    mask = np.zeros([514, 514], np.uint8)
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_GRAY2BGR)
    cv2.floodFill(img_2, mask, (0, 0), (100, 100, 100), (1, 1, 1), (101, 101, 101), cv2.FLOODFILL_FIXED_RANGE)
    img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
    img_2[img_2 == 0] = 255
    img_2[img_2 != 255] = 0

    img_5 = img.copy()
    img_5[img != 0] = 255
    mask = np.zeros([514, 514], np.uint8)
    img_5 = cv2.cvtColor(img_5, cv2.COLOR_GRAY2BGR)
    cv2.floodFill(img_5, mask, (0, 0), (100, 100, 100), (1, 1, 1), (101, 101, 101), cv2.FLOODFILL_FIXED_RANGE)
    img_5 = cv2.cvtColor(img_5, cv2.COLOR_BGR2GRAY)
    img_5[img_5 == 0] = 255
    img_5[img_5 != 255] = 0

    img_fill = np.zeros(img.shape, np.uint8)
    img_fill[img_5 == 255] = 255
    img_fill[img_2 == 255] = 50
    img_fill[img_1 == 255] = 150

    return img_fill


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
    """Polynomial decay of learning rate
    :param init_lr is base learning rate
    :param iter is a current iteration
    :param lr_decay_iter how frequently decay occurs, default is 1
    :param max_iter is number of maximum iterations
    :param power is a polymomial power

    """
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    lr = init_lr * (1 - iter / max_iter) ** power
    optimizer.param_groups[0]['lr'] = lr
    return lr


# return lr

def get_label_info(csv_path, cls_list):
    # return label -> {label_name: [r_value, g_value, b_value, ...}
    ann = pd.read_csv(csv_path)
    label = {}
    for iter, row in ann.iterrows():
        label_name = row['name']
        if label_name in cls_list:
            r = row['r']
            g = row['g']
            b = row['b']
            label[label_name] = [int(r), int(g), int(b)]
    return label


def one_hot_it(label, label_info):
    # return semantic_map -> [H, W, num_classes]
    semantic_map = []
    for info in label_info:
        color = label_info[info]
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and height as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])

    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index
    image = image.permute(1, 2, 0)
    x = torch.argmax(image, dim=-1)
    return x


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """

    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,3])
    # colour_codes = label_values
    # for i in range(0, w):
    #     for j in range(0, h):
    #         x[i, j, :] = colour_codes[int(image[i, j])]
    label_values = [label_values[key] for key in label_values]
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


def compute_global_accuracy(pred, label):
    pred = pred.flatten()
    label = label.flatten()
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)


def compute_dice(pred, label):
    """
    compute dice coefficient
    :param pred: [N, h, w] prob numpy.ndarry
    :param label: [N, h, w] numpy.ndarry
    :return: dice float
    """
    N = pred.shape[0]
    smooth = 0.001

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    pred = pred.reshape((N, -1))
    label = label.reshape((N, -1))

    intersection = pred * label

    dice = (2 * intersection.sum(1) + smooth) / (pred.sum(1) + label.sum(1) + smooth)
    dice = dice.sum() / N

    return dice


def thresh_pred(pred, t_liver, t_tumor):
    """

    :param pred: [N, C, h, w, t]
    :param t_liver:
    :param t_tumor:
    :return:
    """
    bg = pred[:, 0:1]
    liver = pred[:, 1:2]
    tumor = pred[:, 2:3]
    liver[liver >= t_liver] = 1
    liver[liver < t_liver] = 0
    tumor[tumor >= t_tumor] = 1
    tumor[tumor < t_tumor] = 0
    bg[(liver == 0) * (tumor == 0)] = 1
    pred_t = np.concatenate([bg, liver, tumor], axis=1)
    return pred_t


def compute_multi_dice(pred, label):
    """
    compute dice coefficient
    :param pred: [N, C, h, w] prob numpy.ndarry
    :param label: [N, C, h, w] numpy.ndarry
    :return: dice float
    """
    N = pred.shape[0]
    C = pred.shape[1]
    smooth = 0.001
    
    if C == 1:
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        pred = pred[:, 0, ...]
    else:
        pred = np.argmax(pred, axis=1)
    # print(pred)
    # input()
    
    dices = []
    for i in range(C):
        pred_i = pred.copy()
        label_i = label[:, i, :, :, :]
        if C != 1:
            pred_i[pred == i] = 1
            pred_i[pred != i] = 0
        
        pred_i = pred_i.reshape((N, -1))
        label_i = label_i.reshape((N, -1))

        intersection = pred_i * label_i

        dice = (2 * intersection.sum(1) + smooth) / (pred_i.sum(1) + label_i.sum(1) + smooth)

        dice = dice.sum() / N
        dices.append(dice)

    return dices


def compute_multi_hd(pred, label):
    """
    compute dice coefficient
    :param pred: [N, C, h, w (, t)] prob numpy.ndarry
    :param label: [N, C, h, w (, t)] numpy.ndarry
    :return: dice float
    """
    N = pred.shape[0]
    C = pred.shape[1]
    smooth = 0.001

    pred = np.argmax(pred, axis=1)
    # print(pred)
    # input()
    hdcomputer = sitk.HausdorffDistanceImageFilter()


    dices = []
    for i in range(C):
        pred_i = pred.copy()
        label_i = label[:, i, :, :, :]
        pred_i[pred == i] = 1
        pred_i[pred != i] = 0

        pred_i = pred_i.reshape((N, -1))
        label_i = label_i.reshape((N, -1))

        intersection = pred_i * label_i

        dice = (2 * intersection.sum(1) + smooth) / (pred_i.sum(1) + label_i.sum(1) + smooth)

        dice = dice.sum() / N
        dices.append(dice)

    return dices


def compute_multi_dice_tensor(pred, label):
    """
    compute dice coefficient
    :param pred: [N, C, h, w] prob torch.tensor
    :param label: [N, C, h, w] torch.tensor
    :return: dice float
    """
    N = pred.shape[0]
    C = pred.shape[1]
    smooth = 0.001

    pred = torch.argmax(pred, dim=1)
    # print(pred)
    # input()

    dices = []
    for i in range(C):
        pred_i = pred.copy()
        label_i = label[:, i, :, :, :]
        pred_i[pred == i] = 1
        pred_i[pred != i] = 0

        pred_i = pred_i.reshape((N, -1))
        label_i = label_i.reshape((N, -1))

        intersection = pred_i * label_i

        dice = (2 * intersection.sum(1) + smooth) / (pred_i.sum(1) + label_i.sum(1) + smooth)
        dice = dice.sum() / N
        dices.append(dice)

    return dices


def compute_iou(pred, label):
    """
    compute iou coefficient
    :param pred: [N, h, w] prob numpy.ndarry
    :param label: [N, h, w] numpy.ndarry
    :return: dice float
    """
    N = pred.shape[0]
    smooth = 0.001

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    pred = pred.reshape((N, -1))
    label = label.reshape((N, -1))

    intersection = pred * label

    iou = (intersection.sum(1) + smooth) / (pred.sum(1) + label.sum(1) - intersection.sum(1) + smooth)
    iou = iou.sum() / N

    return iou


def compute_multi_iou(pred, label):
    """
    compute iou coefficient
    :param pred: [N, C, h, w] prob numpy.ndarry
    :param label: [N, C, h, w] numpy.ndarry
    :return: iou float
    """
    N = pred.shape[0]
    C = pred.shape[1]
    smooth = 0.001

    pred = np.argmax(pred, axis=1)

    ious = []
    for i in range(C):
        pred_i = pred.copy()
        label_i = label[:, i, :, :]
        pred_i[pred == i] = 1
        pred_i[pred != i] = 0

        pred_i = pred_i.reshape((N, -1))
        label_i = label_i.reshape((N, -1))

        intersection = pred_i * label_i

        iou = (intersection.sum(1) + smooth) / (pred_i.sum(1) + label_i.sum(1) - intersection.sum(1) + smooth)
        iou = iou.sum() / N
        ious.append(iou)

    return ious


def compute_pr(pred, label):
    """
    compute precision coefficient
    :param pred: [N, h, w] prob numpy.ndarry
    :param label: [N, h, w] numpy.ndarry
    :return: dice float
    """
    N = pred.shape[0]
    smooth = 0.001

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    pred = pred.reshape((N, -1))
    label = label.reshape((N, -1))

    tp = pred * label
    fp = pred * (1 - label)

    pr = (tp.sum(1) + smooth) / (tp.sum(1) + fp.sum(1) + smooth)
    pr = pr.sum() / N

    return pr


def compute_rc(pred, label):
    """
    compute recall coefficient
    :param pred: [N, h, w] prob numpy.ndarry
    :param label: [N, h, w] numpy.ndarry
    :return: dice float
    """
    N = pred.shape[0]
    smooth = 0.001

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    pred = pred.reshape((N, -1))
    label = label.reshape((N, -1))

    tp = pred * label
    fn = (1 - pred) * label

    rc = (tp.sum(1) + smooth) / (tp.sum(1) + fn.sum(1) + smooth)
    rc = rc.sum() / N

    return rc


def compute_spe(pred, label):
    """
    compute specificity coefficient
    :param pred: [N, h, w] prob numpy.ndarry
    :param label: [N, h, w] numpy.ndarry
    :return: dice float
    """
    N = pred.shape[0]
    smooth = 0.001

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    pred = pred.reshape((N, -1))
    label = label.reshape((N, -1))

    tn = (1 - pred) * (1 - label)
    fp = pred * (1 - label)

    spe = (tn.sum(1) + smooth) / (tn.sum(1) + fp.sum(1) + smooth)
    spe = spe.sum() / N

    return spe


def compute_f1(pred, label):
    """
    compute f1-score coefficient
    :param pred: [N, h, w] prob numpy.ndarry
    :param label: [N, h, w] numpy.ndarry
    :return: dice float
    """
    pr = compute_pr(pred, label)
    rc = compute_rc(pred, label)
    f1 = (2 * pr * rc) / (pr + rc)

    return f1


def multi2binary(pred):
    """
    :param pred: [N, C, h, w] prob numpy.ndarry
    :param label: [N, C, h, w] numpy.ndarry
    :return: pred: [N, h, w], label: [N, h, w]
    """
    pred = np.argmax(pred, axis=1)
    pred[pred != 0] = 1
    return pred


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, target, input, W=None):
        N = target.size(0)
        smooth = 0.001

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, target, input, W=None):

        C = target.shape[1]

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if W is not None:
                diceLoss *= W[i]
            totalLoss += diceLoss

        if W is not None:
            return totalLoss / sum(W)
        else:
            return totalLoss / (C)


class XentLoss(nn.Module):
    def __init__(self, bWeighted=False, gamma=0, bMask=False):
        """
        categorical (multi-class) cross-entropy loss, capable of adding weight, focal loss.
        input
            bWeighted: True if need weighted loss
            gamma: if >0, becomes focal loss
            bMask: if calculate loss in mask area
        """
        super(XentLoss, self).__init__()
        self.bWeighted = bWeighted
        self.bFL = gamma > 0
        self.gamma = gamma
        self.bMask = bMask
        self.channel_axis = 1

    def forward(self, lb, pred, W=None, mask=None):
        """
        input
            lb: true label, assume to be one-hot format.
            pred: prediction
            W: weight of each class
            mask: one channel tensor of [batch, H, W], elements must be {1,0}
        """
        if self.bMask or self.bWeighted:
            sz = lb.data.shape

        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
        loss = - lb * torch.log(pred)

        # use focal loss
        if self.bFL:  # if use focal loss
            loss = loss * (1 - pred) ** self.gamma
        loss = torch.sum(loss, self.channel_axis)

        # use mask to focus on part of loss
        if self.bMask:
            loss = loss * mask

        # weight the loss
        if self.bWeighted:
            wm = torch.zeros(sz).cuda()
            for k in range(sz[self.channel_axis]):  # for each channel
                wm[:, k, :, :] = lb[:, k, :, :] * W[k]
            loss = loss * torch.sum(wm, self.channel_axis)
        return torch.mean(loss)


def imgs2arrays(root, img_dict):
    array_dict = dict()
    for key, values in img_dict.items():
        values.sort()
        gt_arrays = []
        pred_arrays = []
        for name in values:
            img_path = os.path.join(root, name)
            img = cv2.imread(img_path)
            gt_arrays.append(img[:, :512, 0])
            pred_arrays.append(img[:, 513:, 0])
        gt_arrays = np.concatenate(gt_arrays, axis=0)[np.newaxis, :, :]
        pred_arrays = np.concatenate(pred_arrays, axis=0)[np.newaxis, :, :]
        array_dict[key] = [gt_arrays, pred_arrays]
    return array_dict


def arrays2onehot(arrays):
    """
    :param arrays: [1, n*h, w]
    :return: onehot arrays: [1, c, n*h, w]
    """
    arrays[arrays == 255] = 1
    arrays[arrays == 150] = 2
    arrays[arrays == 50] = 2
    arrays = np.eye(3)[arrays]
    arrays = np.transpose(arrays, [0, 3, 1, 2])
    return arrays


def compute_sample_dice(img_arrays):
    bg_list, liver_list, tumor_list = [], [], []
    for arrays in img_arrays.values():
        gt, pred = arrays
        gt = arrays2onehot(gt)
        pred = arrays2onehot(pred)
        bg, liver, tumor = compute_multi_dice(pred, gt)
        print(bg)
        print(liver)
        print(tumor)
        bg_list.append(bg)
        liver_list.append(liver)
        tumor_list.append(tumor)
    return np.mean(bg_list), np.mean(liver_list), np.mean(tumor_list)


def dice_from_imgs(root):
    img_dict = dict()
    names = os.listdir(root)
    for name in names:
        id = name[:-7]
        if id in img_dict:
            img_dict[id].append(name)
        else:
            img_dict[id] = []
            img_dict[id].append(name)

    img_arrays = imgs2arrays(root, img_dict)
    bg_dice, liver_dice, tumor_dice = compute_sample_dice(img_arrays)
    print(bg_dice)
    print(liver_dice)
    print(tumor_dice)


if __name__ == '__main__':
    dice_from_imgs('../liver_data/preds')
