import torch
import os
from torchvision import transforms
from torchvision.transforms import functional as F
import cv2
from PIL import Image
import numpy as np
from imgaug import augmenters as iaa
import imgaug as ia
import sys

sys.path.append('..')
from utils import get_label_info, one_hot_it
# from utils import *
import random


def augmentation():
	# augment images with spatial transformation: Flip, Affine, Rotation, etc...
	# see https://github.com/aleju/imgaug for more details
	pass


def augmentation_pixel():
	# augment images with pixel intensity transformation: GaussianBlur, Multiply, etc...
	pass


def load_img(path, resize_func):
	img = cv2.imread(path)
	img = Image.fromarray(img)
	img = resize_func(img)
	img = np.array(img)
	return img


def ajacent_img(img_prefix, ind, offset, resize_func):
	img_aj_path = img_prefix + '{}.png'.format(ind + offset)
	if os.path.exists(img_aj_path):
		return load_img(img_aj_path, resize_func)
	else:
		offset = offset + 1 if offset < 0 else offset - 1
		return ajacent_img(img_prefix, ind, offset, resize_func)


class Thoracic(torch.utils.data.Dataset):
	def __init__(self, image_list, scale, cls_list, mode='train', img_seq=False, len_seq=3,
				 aug=True, hard=False, pred=False, tta=False):
		super().__init__()
		self.mode = mode
		self.aug = aug
		self.scale = scale
		self.cls_list = cls_list
		self.pred = pred
		self.tta = tta
		self.img_seq = img_seq
		self.len_seq = len_seq
		
		with open(image_list, 'r') as f:
			image_list = f.readlines()
		
		image_list = [line[:-1] for line in image_list]
		
		if hard and (mode == 'train'):
			hard_id = []
			with open('./data_split/hard.txt', 'r') as f:
				lines = f.readlines()
			for line in lines:
				hard_id.append(int(line[:-1]))
			hard_list = [name for name in image_list if int(name.split('/')[-2]) in hard_id]
			image_list.extend(hard_list)
		
		self.resize_label = transforms.Resize(scale, Image.NEAREST)
		self.resize_img = transforms.Resize(scale, Image.BILINEAR)
		
		self.image_list = image_list
		self.num_classes = len(cls_list)
		
		if self.img_seq:
			ch_num = self.len_seq
		else:
			ch_num = 3
		self.to_tensor = transforms.Compose([
			transforms.ToTensor(),
			# transforms.Normalize(mean=[0.62335] * ch_num, std=[0.24169] * ch_num), # lung's
			# transforms.Normalize(mean=[0.22162] * ch_num, std=[0.27966] * ch_num), # esophagus's mean and std
			# transforms.Normalize(mean=[0.37639] * ch_num, std=[0.29705] * ch_num),  # esophagus_single's mean and std
			transforms.Normalize(mean=[0.64361] * ch_num, std=[0.34342] * ch_num),  # trachea's mean and std
			# transforms.Normalize(mean=[0.30844] * ch_num, std=[0.19496] * ch_num),  # all's mean and std
		])
		
		# self.seq = iaa.Noop()
		self.seq = iaa.Sequential([
			iaa.Crop(px=(0, 10)),
			# iaa.Flipud(0.5),
			iaa.SomeOf((0, 5), [
				iaa.Sometimes(0.5, iaa.Affine(
					scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
					rotate=(-10, 10),
					shear=(-10, 10),
				)),
				iaa.PiecewiseAffine(scale=(0.01, 0.05)),
				iaa.GaussianBlur(sigma=(0, 2.0)), #trachea use 2.0
				iaa.Multiply((0.6, 1.2)),   #lung's
			# 	# iaa.Multiply((0.4, 1.4)),   #esophagus's
			# 	iaa.Multiply((0.6, 1.2)),  # trachea's
			# 	# iaa.PerspectiveTransform(scale=(0, 0.1)),
			# 	# iaa.Superpixels(p_replace=(0, 0.3)),
			])
		])
	
	def __getitem__(self, index):
		
		name = self.image_list[index]

		pid = name.split('/')[-2]
		sid = name.split('/')[-1][:-4]
		
		tta_seq = []
		
		if self.pred:
			pred_npy_path = '/data/lbw/structseg2019/data/pred_trachea_npy/pred_{}.npy'.format(pid)
			pred = np.load(pred_npy_path)[0, 1:, :, :, int(sid)]
			img = cv2.imread(self.image_list[index])[:, :, 0]
			img = np.stack([img, pred[0]], axis=2)
			img = Image.fromarray(img)
			img = self.resize_img(img)
			img = np.array(img)
		else:
			img = load_img(self.image_list[index], self.resize_img)
		
		label = cv2.imread(self.image_list[index].replace('img', 'label'), 0)
		
		organs = ['Bg', 'RightLung', 'LeftLung', 'Heart', 'Esophagus', 'Trachea', 'SpinalCord']
		
		cnt = 0
		for i, organ in enumerate(organs):
			if organ in self.cls_list:
				label[label == i] = cnt
				cnt += 1
			else:
				label[label == i] = 0
		
		label_full = np.eye(self.num_classes)[label]
		label_full[:, :, 0] = 1 - label_full[:, :, 1:].sum(axis = 2)
		
		label = Image.fromarray(label)
		label = self.resize_label(label)
		label = np.eye(self.num_classes)[label]
		
		if self.img_seq:
			path_pre = self.image_list[index].split('img')[0] + 'img/'
			img_prefix = path_pre + self.image_list[index].split('/')[-2] + '/'
			img_ind = int(self.image_list[index].split('/')[-1][:-4])
			offset_range = [ind - (self.len_seq - 1) // 2 for ind in range(self.len_seq)]
			img_list = [ajacent_img(img_prefix, img_ind, offset, self.resize_img) for offset in offset_range]
			img = np.stack([img_aj[:, :, 0] for img_aj in img_list], axis=2)
		
		
		if self.mode == 'train' and self.aug:
			seq_det = self.seq.to_deterministic()
			label = np.argmax(label, axis=2).astype(np.int8)
			segmap = ia.SegmentationMapOnImage(label, shape=img.shape, nb_classes=self.num_classes)
			img = seq_det.augment_image(img)
			label = seq_det.augment_segmentation_maps([segmap])[0]
			vis_path = '/data/ybh/PublicDataSet/StructSeg2019/vis_temp/'
			cv2.imwrite(vis_path + str(index) + '_img.png', img)
			cv2.imwrite(vis_path + str(index) + '_mask.png', label.draw_on_image(img))
			
			
			label = np.eye(self.num_classes)[label.get_arr_int()]
		elif self.tta:
			crop = random.randint(0, 10)
			scale_x = random.random() * 0.4 + 0.8
			scale_y = random.random() * 0.4 + 0.8
			rotate = random.randint(-10, 10)
			tta_seq.extend([crop, scale_x, scale_y, rotate])
			seq_test = iaa.Sequential([
				iaa.Crop(px=crop),
				iaa.Affine(
					scale={"x": scale_x, "y": scale_y},
					rotate=rotate,
				),
				iaa.GaussianBlur(sigma=(0, 3.0)),
				iaa.Multiply((0.4, 1.4)),
				# iaa.PerspectiveTransform(scale=(0, 0.1)),
				# iaa.Superpixels(p_replace=(0, 0.3)),
				])

			seq_test_det = seq_test.to_deterministic()
			label = np.argmax(label, axis=2)
			segmap = ia.SegmentationMapOnImage(label, shape=img.shape, nb_classes=self.num_classes)
			img = seq_test_det.augment_image(img)
			label = seq_test_det.augment_segmentation_maps([segmap])[0]

			label = np.eye(self.num_classes)[label.get_arr_int()]
		
		# print(img.shape)
		img = self.to_tensor(img.astype(np.uint8)).float()
		
		label = np.transpose(label, [2, 0, 1]).astype(np.float32)
		label = torch.from_numpy(label)[:, :, :]
		
		label_full = np.transpose(label_full, [2, 0, 1]).astype(np.float32)
		label_full = torch.from_numpy(label_full)[:, :, :]
		
		if self.mode == 'train':
			return img, label
		elif self.tta:
			return img, label_full, name, tta_seq
		else:
			return img, label_full, name
		
	def __len__(self):
		return 10
		# return len(self.image_list)


# class ThoracicCLS(torch.utils.data.Dataset):
#     def __init__(self, image_list, scale, cls_list, mode='train', aug=True, img_seq=False, len_seq=3):
#         super().__init__()
#         self.mode = mode
#         self.aug = aug
#         self.img_seq = img_seq
#         self.len_seq = len_seq
#         self.scale = scale
#         self.path_pre = '/data/ybh/PublicDataSet/StructSeg2019/Thoracic_OAR_crop/img/'
#         self.cls_list = cls_list
#         # self.cnt = 0
#
#         with open(image_list, 'r') as f:
#             image_list = f.readlines()
#         # image_list = [line[:-1].replace('trachea_data', 'trachea_data_contrast') for line in image_list]
#         image_list = [line[:-1].replace('ybh/PublicDataSet/StructSeg2019/Thoracic_OAR_crop', 'lbw/structseg2019/data_ori/crop_data_L35W200') for line in image_list]
#         # image_list = [line[:-1] for line in image_list]
#
#         self.resize_label = transforms.Resize(scale, Image.NEAREST)
#         self.resize_img = transforms.Resize(scale, Image.BILINEAR)
#
#         self.image_list = image_list
#         self.to_tensor = transforms.ToTensor()
#         self.seq = iaa.Sequential([
#             iaa.Crop(px=(0, 10)),
#             # iaa.Flipud(0.5),
#             iaa.SomeOf((0, 5), [
#                 iaa.Sometimes(0.5, iaa.Affine(
#                     scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
#                     rotate=(-10, 10),
#                 )),
#                 iaa.GaussianBlur(sigma=(0, 3.0)),
#                 iaa.Multiply((0.4, 1.4)),
#                 # iaa.PerspectiveTransform(scale=(0, 0.1)),
#                 # iaa.Superpixels(p_replace=(0, 0.3)),
#             ])
#         ])
#
#     def __getitem__(self, index):
#
#         img = load_img(self.image_list[index], self.resize_img)
#         label = cv2.imread(self.image_list[index].replace('img', 'label'))
#
#         organs = ['Bg', 'RightLung', 'LeftLung', 'Heart', 'Trachea', 'Esophagus', 'SpinalCord']
#         label = [1 if (label==organs.index(name)).any() else 0 for name in self.cls_list if name != 'Bg']
#         label = torch.from_numpy(np.float32(label))
#
#         name = self.image_list[index]
#
#         if self.img_seq:
#             img_prefix = self.path_pre + self.image_list[index].split('/')[-2] + '/'
#             img_ind = int(self.image_list[index].split('/')[-1][:-4])
#             offset_range = [ind - (self.len_seq - 1) // 2 for ind in range(self.len_seq)]
#             img_list = [ajacent_img(img_prefix, img_ind, offset, self.resize_img) for offset in offset_range]
#             img = np.stack([img_aj[:, :, 0] for img_aj in img_list], axis=2)
#
#         if self.mode == 'train' and self.aug:
#             seq_det = self.seq.to_deterministic()
#             img = seq_det.augment_image(img)
#
#         # print(img.shape)
#         img = self.to_tensor(img.astype(np.uint8)).float()
#
#         return img, label
#
#     def __len__(self):
#         return len(self.image_list)


if __name__ == '__main__':
	# dataset_val = Liver('../data_split/lits_val.txt', 'lits', '../class_dict.csv', (224, 224), num_classes=3,
	# 					mode='val', aug=False, img_seq=True, in_liver=False, len_seq=5)

	# dataset_val = Thoracic('../data_split/trachea/train_fold1.txt', '../class_dict.csv', (128, 128), ['Bg', 'Trachea'], mode='train', aug=True,
	# 					   img_seq=False, in_range=False, len_seq=5, pred=True)
	
	# organs = ['Bg', 'Trachea']
	
	organs = ['Bg', 'RightLung', 'LeftLung', 'Heart', 'Esophagus', 'Trachea', 'SpinalCord']
	
	dataset_val = Thoracic('/data/ybh/Programs/StructSeg2019_2d/data_split/lung/val_one.txt', (207, 143), organs, img_seq=False, len_seq=5,
	                            mode='train', aug=True, hard=False, pred=False)
	
	# dataset_val = Thoracic('../2d_data_split/val_lung.txt', (256, 256), ['Bg', 'RightLung', 'LeftLung'],
	#                        mode='train', aug=True, hard=False, pred=False)

	# dataset_train = Thoracic(args.train_path, args.csv_path, scale=(args.crop_h, args.crop_w), mode='train', aug=True,
	#                          img_seq=seq, in_range=args.in_range, len_seq=args.len_seq, hard=False)
	print('size of valset', len(dataset_val))
	
	
	for i, (img, label) in enumerate(dataset_val):
		# print(img)
		# print(label)
		print(i, img.shape, label.shape)
		# print(img.shape)
		# print(label.shape)
		# input()
		# patient_id = name.split('/')[-2]
		# slice_id = name.split('/')[-1][:-4]
		# print(patient_id, slice_id)
		if i > 498:
			break
# input()

