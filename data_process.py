import os
import random
import shutil
import cv2
import numpy as np
import SimpleITK as sitk
import random
from utils import *
from nibabel import load as load_nii
import nibabel as nib
import json
# import nrrd


def read_nii(path, winwidth, wincenter):
	img = load_nii(path).get_data()
	img = setDicomWinWidthWinCenter(img, winwidth, wincenter) # liver 300 -20
	img = np.uint8(img)
	return img


def nii2img(nii_root, target_root):
	names = [name for name in os.listdir(nii_root) if 'volume' in name]
	if not os.path.exists(target_root):
		os.mkdir(target_root)

	for i, name in enumerate(names):
		nii_path = os.path.join(nii_root, name)
		target_path = os.path.join(target_root, name[:-4])
		if not os.path.exists(target_path):
			os.mkdir(target_path)

		image_array = read_nii(nii_path)
		print(i, image_array.shape)
		for i in range(image_array.shape[2]):
			cv2.imwrite(os.path.join(target_path, '{:0>3d}.png'.format(i+1)), image_array[:, :, i])


def nii2label(nii_root, target_root):
	names = [name for name in os.listdir(nii_root) if 'segmentation' in name]
	if not os.path.exists(target_root):
		os.mkdir(target_root)

	for i, name in enumerate(names):
		nii_path = os.path.join(nii_root, name)
		target_path = os.path.join(target_root, name[:-4])
		if not os.path.exists(target_path):
			os.mkdir(target_path)

		label_array = np.uint8(load_nii(nii_path).get_data())
		label_array[label_array == 1] = 255
		label_array[label_array == 2] = 150
		print(i, label_array.shape)
		for i in range(label_array.shape[2]):
			cv2.imwrite(os.path.join(target_path, '{:0>3d}.png'.format(i+1)), label_array[:, :, i])


def dcm2npy(dcm_root, target_imgs_root, target_labels_root):
	dirs = os.listdir(dcm_root)
	if not os.path.exists(target_imgs_root):
		os.mkdir(target_imgs_root)

	for dir in dirs:
		dir_path = os.path.join(dcm_root, dir)

		names = [name for name in os.listdir(dir_path) if 'CT' in name]
		ct_path = os.path.join(dir_path, names[0])

		# HA img
		ha_path = os.path.join(ct_path, 'HA')
		target_ha_path = os.path.join(target_imgs_root, dir + '_HA')
		image_array = readDcmSeries(ha_path)
		print(image_array.shape)
		np.save(target_ha_path, image_array)

		# HA label
		mha_names = [name for name in os.listdir(ha_path) if '.mha' in name]
		ha_mha_path = os.path.join(ha_path, mha_names[0])
		target_ha_label_path = os.path.join(target_labels_root, dir + '_HA')
		image = sitk.ReadImage(ha_mha_path)
		label_array = sitk.GetArrayFromImage(image)
		label_npy = np.stack([fill_edge(img) for img in label_array], axis=0)
		print(label_npy.shape)
		np.save(target_ha_label_path, label_npy)

		# PV img
		pv_path = os.path.join(ct_path, 'PV')
		target_pv_path = os.path.join(target_imgs_root, dir + '_PV')
		image_array = readDcmSeries(pv_path)
		print(image_array.shape)
		np.save(target_pv_path, image_array)

		# PV label
		mha_names = [name for name in os.listdir(pv_path) if '.mha' in name]
		pv_mha_path = os.path.join(pv_path, mha_names[0])
		target_pv_label_path = os.path.join(target_labels_root, dir + '_PV')
		image = sitk.ReadImage(pv_mha_path)
		label_array = sitk.GetArrayFromImage(image)
		label_npy = np.stack([fill_edge(img) for img in label_array], axis=0)
		print(label_npy.shape)
		np.save(target_pv_label_path, label_npy)

		# input()


def dcm2img(dcm_root, target_root):
	dirs = os.listdir(dcm_root)
	if not os.path.exists(target_root):
		os.mkdir(target_root)

	for dir in dirs:
		dir_path = os.path.join(dcm_root, dir)
		target_path = os.path.join(target_root, dir)
		if not os.path.exists(target_path):
			os.mkdir(target_path)

		names = [name for name in os.listdir(dir_path) if 'CT' in name]
		ct_path = os.path.join(dir_path, names[0])

		ha_path = os.path.join(ct_path, 'HA')
		target_ha_path = os.path.join(target_path, 'HA')
		if not os.path.exists(target_ha_path):
			os.mkdir(target_ha_path)
		image_array = readDcmSeries(ha_path)
		print(image_array.shape)
		for i in range(len(image_array)):
			cv2.imwrite(os.path.join(target_ha_path, '{:0>3d}.png'.format(i+1)), image_array[i])

		pv_path = os.path.join(ct_path, 'PV')
		target_pv_path = os.path.join(target_path, 'PV')
		if not os.path.exists(target_pv_path):
			os.mkdir(target_pv_path)
		image_array = readDcmSeries(pv_path)
		print(image_array.shape)
		for i in range(len(image_array)):
			cv2.imwrite(os.path.join(target_pv_path, '{:0>3d}.png'.format(i + 1)), image_array[i])


def mha2label(root, target_root):
	dirs = os.listdir(root)
	if not os.path.exists(target_root):
		os.mkdir(target_root)

	for dir in dirs:
		print(dir)
		dir_path = os.path.join(root, dir)
		target_path = os.path.join(target_root, dir)
		# img_ha_path = os.path.join('../liver_data/imgs', dir, 'HA')
		# img_pv_path = os.path.join('../liver_data/imgs', dir, 'PV')
		if not os.path.exists(target_path):
			os.mkdir(target_path)

		names = [name for name in os.listdir(dir_path) if 'CT' in name]
		ct_path = os.path.join(dir_path, names[0])

		ha_path = os.path.join(ct_path, 'HA')
		names = [name for name in os.listdir(ha_path) if '.mha' in name]
		ha_mha_path = os.path.join(ha_path, names[0])
		target_ha_path = os.path.join(target_path, 'HA')
		if not os.path.exists(target_ha_path):
			os.mkdir(target_ha_path)
		image = sitk.ReadImage(ha_mha_path)
		image_array = sitk.GetArrayFromImage(image)
		image_array[(image_array != 0) * (image_array != 1) * (image_array != 5)] = 1
		for i in range(len(image_array)):
			# img = image_array[i]
			# a = img[(img != 0) * (img != 1) * (img != 2)]
			# if len(a) != 0:
			#     print(a)
			#     input()

			# # img = fill_edge(image_array[i])
			# # print(img[(img!=0) * (img!=50) * (img!=150) * (img!=255)])

			# target_ha_path = target_root
			cv2.imwrite(os.path.join(target_ha_path, '{:0>3d}.png'.format(i + 1)), fill_edge(image_array[i]))
			# # img = cv2.imread(os.path.join(target_ha_path, '{:0>3d}.png'.format(i + 1)))
			# # print(img[(img != 0) * (img != 50) * (img != 150) * (img != 255)])

		pv_path = os.path.join(ct_path, 'PV')
		names = [name for name in os.listdir(pv_path) if '.mha' in name]
		pv_mha_path = os.path.join(pv_path, names[0])
		target_pv_path = os.path.join(target_path, 'PV')
		if not os.path.exists(target_pv_path):
			os.mkdir(target_pv_path)
		image = sitk.ReadImage(pv_mha_path)
		image_array = sitk.GetArrayFromImage(image)
		image_array[(image_array != 0) * (image_array != 1) * (image_array != 5)] = 1
		for i in range(len(image_array)):
			# img = image_array[i]
			# a = img[(img != 0) * (img != 1) * (img != 2)]
			# if len(a) != 0:
			#     print(a)
			#     input()

			# target_pv_path = target_root
			cv2.imwrite(os.path.join(target_pv_path, '{:0>3d}.png'.format(i + 1)), fill_edge(image_array[i]))


def mha2nii(root, target_root):
	dirs = os.listdir(root)
	if not os.path.exists(target_root):
		os.mkdir(target_root)

	for dir in dirs:
		print(dir)
		dir_path = os.path.join(root, dir)

		names = [name for name in os.listdir(dir_path) if 'CT' in name]
		ct_path = os.path.join(dir_path, names[0])

		ha_path = os.path.join(ct_path, 'HA')
		names = [name for name in os.listdir(ha_path) if '.mha' in name]
		ha_mha_path = os.path.join(ha_path, names[0])
		ha_save_name = ha_mha_path.split('/')[-4] + '_' + ha_mha_path.split('/')[-2]
		print(ha_save_name)
		ha_save_path = os.path.join(target_root, ha_save_name)
		image = sitk.ReadImage(ha_mha_path)
		image_array = sitk.GetArrayFromImage(image)
		image_array[(image_array != 0) * (image_array != 1) * (image_array != 5)] = 1
		for i in range(len(image_array)):
			image_array[i] = fill_edge(image_array[i])
		image_array[image_array == 255] = 1
		image_array[image_array == 150] = 2
		image_array = image_array.astype(np.uint8).transpose([2, 1, 0])

		image_array = nib.Nifti1Image(image_array, None)
		nib.save(image_array, ha_save_path)

		pv_path = os.path.join(ct_path, 'PV')
		names = [name for name in os.listdir(pv_path) if '.mha' in name]
		pv_mha_path = os.path.join(pv_path, names[0])
		pv_save_name = pv_mha_path.split('/')[-4] + '_' + pv_mha_path.split('/')[-2]
		print(pv_save_name)
		pv_save_path = os.path.join(target_root, pv_save_name)
		image = sitk.ReadImage(pv_mha_path)
		image_array = sitk.GetArrayFromImage(image)
		image_array[(image_array != 0) * (image_array != 1) * (image_array != 5)] = 1
		for i in range(len(image_array)):
			image_array[i] = fill_edge(image_array[i])
		image_array[image_array == 255] = 1
		image_array[image_array == 150] = 2
		image_array = image_array.astype(np.uint8).transpose([2, 1, 0])

		image_array = nib.Nifti1Image(image_array, None)
		nib.save(image_array, pv_save_path)


def split_data_npy(name_root, root, target):
	if not os.path.exists(target):
		os.mkdir(target)
	train_path = os.path.join(target, 'train_3d.txt')
	val_path = os.path.join(target, 'val_3d.txt')
	test_path = os.path.join(target, 'test_3d.txt')

	names = os.listdir(name_root)
	random.shuffle(names)

	with open(train_path, 'w') as f:
		f.write('')
	for name in names[:62]:
		ha_path = os.path.join(root, name + '_HA.npy')
		pv_path = os.path.join(root, name + '_PV.npy')
		with open(train_path, 'a') as f:
			f.write(ha_path + '\r\n')
			f.write(pv_path + '\r\n')

	with open(val_path, 'w') as f:
		f.write('')
	for name in names[62:82]:
		ha_path = os.path.join(root, name + '_HA.npy')
		pv_path = os.path.join(root, name + '_PV.npy')
		with open(val_path, 'a') as f:
			f.write(ha_path + '\r\n')
			f.write(pv_path + '\r\n')

	with open(test_path, 'w') as f:
		f.write('')
	for name in names[82:102]:
		ha_path = os.path.join(root, name + '_HA.npy')
		pv_path = os.path.join(root, name + '_PV.npy')
		with open(test_path, 'a') as f:
			f.write(ha_path + '\r\n')
			f.write(pv_path + '\r\n')


# def split_data(root, target):
#     if not os.path.exists(target):
#         os.mkdir(target)
#     train_path = os.path.join(target, 'train.txt')
#     val_path = os.path.join(target, 'val.txt')
#     test_path = os.path.join(target, 'test.txt')
#
#     imgs_path = os.path.join(root, 'imgs')
#     names = os.listdir(imgs_path)
#     random.shuffle(names)
#
#     with open(train_path, 'w') as f:
#         f.write('')
#     for name in names[:62]:
#         ha_path = os.path.join(imgs_path, name, 'HA')
#         for i in range(len(os.listdir(ha_path))):
#             img_path = os.path.join(ha_path, '{:0>3d}.png'.format(i+1))
#             with open(train_path, 'a') as f:
#                 f.write(img_path + '\r\n')
#         pv_path = os.path.join(imgs_path, name, 'PV')
#         for i in range(len(os.listdir(pv_path))):
#             img_path = os.path.join(pv_path, '{:0>3d}.png'.format(i + 1))
#             with open(train_path, 'a') as f:
#                 f.write(img_path + '\r\n')
#
#     with open(val_path, 'w') as f:
#         f.write('')
#     for name in names[62:82]:
#         ha_path = os.path.join(imgs_path, name, 'HA')
#         for i in range(len(os.listdir(ha_path))):
#             img_path = os.path.join(ha_path, '{:0>3d}.png'.format(i + 1))
#             with open(val_path, 'a') as f:
#                 f.write(img_path + '\r\n')
#         pv_path = os.path.join(imgs_path, name, 'PV')
#         for i in range(len(os.listdir(pv_path))):
#             img_path = os.path.join(pv_path, '{:0>3d}.png'.format(i + 1))
#             with open(val_path, 'a') as f:
#                 f.write(img_path + '\r\n')
#
#     with open(test_path, 'w') as f:
#         f.write('')
#     for name in names[82:102]:
#         ha_path = os.path.join(imgs_path, name, 'HA')
#         for i in range(len(os.listdir(ha_path))):
#             img_path = os.path.join(ha_path, '{:0>3d}.png'.format(i + 1))
#             with open(test_path, 'a') as f:
#                 f.write(img_path + '\r\n')
#         pv_path = os.path.join(imgs_path, name, 'PV')
#         for i in range(len(os.listdir(pv_path))):
#             img_path = os.path.join(pv_path, '{:0>3d}.png'.format(i + 1))
#             with open(test_path, 'a') as f:
#                 f.write(img_path + '\r\n')


def split_data_lits(root, target_path):
	names = os.listdir(root)
	random.seed = 0
	random.shuffle(names)

	with open(os.path.join(target_path, 'lits_train.txt'), 'w') as f:
		for name in names[:100]:
			f.write(name + '\r\n')

	with open(os.path.join(target_path, 'lits_val.txt'), 'w') as f:
		for name in names[100:]:
			f.write(name + '\r\n')


def locate_z(root, target):
	names = [name for name in os.listdir(root) if ('segmentation' in name) or ('ori' in name)]
	with open(target, 'w') as f:
		f.write('')

	for i, name in enumerate(names):
		print(i + 1)
		nii_path = os.path.join(root, name)

		label_array = np.uint8(load_nii(nii_path).get_data())
		label_array[label_array == 2] = 1
		z = np.where(label_array == 1)[2]
		len_z = label_array.shape[2]
		name = 'test-volume-' + name[:-8] + '.nii'
		with open(target, 'a') as f:
			f.write('{} {} {} {}\r\n'.format(name, np.min(z), np.max(z), len_z))


def locate_zheyi_tumor(root, target):
	names = os.listdir(root)
	with open(target, 'w') as f:
		f.write('')

	for i, name in enumerate(names):
		print(name)
		nii_path = os.path.join(root, name)

		label_array = np.uint8(load_nii(nii_path).get_data())
		z = np.where(label_array == 2)[2]
		if len(z) == 0:
			with open(target, 'a') as f:
				f.write('{} {} {}\r\n'.format(name, -1, -1))
			continue
		with open(target, 'a') as f:
			f.write('{} {} {}\r\n'.format(name, np.min(z), np.max(z)))


def data_distribution(root):
	names = os.listdir(root)
	name_paths = [os.path.join(root, name) for name in names]
	dir_paths = []
	for name_path in name_paths:
		dir_paths.append(os.path.join(name_path, 'HA'))
		dir_paths.append(os.path.join(name_path, 'PV'))

	cnt, cnt_liver, cnt_tumor = 0, 0, 0
	for path in dir_paths:
		label_names = os.listdir(path)
		for label_name in label_names:
			label_path = os.path.join(path, label_name)
			label = cv2.imread(label_path)
			if (label == 255).any():
				cnt_liver += 1
			if (label == 150).any():
				cnt_tumor += 1
			cnt += 1

	print('num of slices', cnt)  # 10024
	print('num of bg slices', cnt - cnt_liver)  # 3621
	print('num of liver slices', cnt_liver)  # 6403
	print('num of tumor slices', cnt_tumor)  # 1854


def max_range():
	root = '/data/lbw/structseg2019/data_ori/Task3_Thoracic_OAR'
	paths = ['{}/{}/label.nii.gz'.format(root, i) for i in os.listdir(root)]
	min_h, max_h, min_w, max_w = 1000, -1, 1000, -1

	for path in paths:
		label = load_nii(path).get_data().astype(np.uint8)
		h, w, t = np.where(label != 0)
		print('min h: {} | max h: {} | min w: {} | max w: {}'.format(min(h), max(h), min(w), max(w)))
		min_h = min(min_h, min(h))
		max_h = max(max_h, max(h))
		min_w = min(min_w, min(w))
		max_w = max(max_w, max(w))

	print('all')
	print('min h: {} | max h: {} | min w: {} | max w: {}'.format(min_h, max_h, min_w, max_w))


def z_range():
	root = '/data/lbw/structseg2019/data_ori/Task3_Thoracic_OAR'
	paths = ['{}/{}/label.nii.gz'.format(root, i) for i in os.listdir(root)]
	min_h, max_h, min_w, max_w = 1000, -1, 1000, -1

	for path in paths:
		label = load_nii(path).get_data().astype(np.uint8)
		rl_h, rl_w, rl_t = np.where(label == 1)
		ll_h, ll_w, ll_t = np.where(label == 2)
		tr_h, tr_w, tr_t = np.where(label == 5)
		print(path)
		print(label.shape)
		print('min lung_t: {} | max lung_t: {} | min tra_t: {} | max tra_t: {}'.format(min(min(rl_t), min(ll_t)), max(max(rl_t), max(ll_t)), min(tr_t), max(tr_t)))
		# min_h = min(min_h, min(h))
		# max_h = max(max_h, max(h))
		# min_w = min(min_w, min(w))
		# max_w = max(max_w, max(w))

	# print('all')
	# print('min h: {} | max h: {} | min w: {} | max w: {}'.format(min_h, max_h, min_w, max_w))


def dump_range():
	root = '/data/lbw/structseg2019/data_ori/Task3_Thoracic_OAR'
	label_range = dict()
	organs = ['Bg', 'RightLung', 'LeftLung', 'Heart', 'Trachea', 'Esophagus', 'SpinalCord']

	for i in range(1, 51):
		print(i)
		path = '{}/{}/label.nii.gz'.format(root, i)
		label_range[i] = dict()
		label = load_nii(path).get_data().astype(np.uint8)
		label_range[i]['shape'] = (int(label.shape[0]), int(label.shape[1]), int(label.shape[2]))

		for j in range(1, len(organs)):
			h, w, t = np.where(label == j)
			label_range[i][organs[j]] = (int(min(h)), int(min(w)), int(min(t)), int(max(h)), int(max(w)), int(max(t)))

	with open('./data_split/label_range.json', 'w') as f:
		json.dump(label_range, f)


def dump_spacing():
	root = '/data/lbw/structseg2019/data_ori/Task3_Thoracic_OAR'
	img_range = dict()

	for i in range(1, 51):
		print(i)
		path = '{}/{}/data.nii.gz'.format(root, i)
		img_range[i] = dict()
		img = load_nii(path).get_data().astype(np.uint8)
		spacing = load_nii(path).header['pixdim'][1:4]
		img_range[i]['shape'] = (int(img.shape[0]), int(img.shape[1]), int(img.shape[2]))
		img_range[i]['spacing'] = (float(spacing[0]), float(spacing[1]), float(spacing[2]))

	with open('./data_split/img_spacing.json', 'w') as f:
		json.dump(img_range, f)


def generate_data(mode):
	root = '/data/lbw/structseg2019/data_ori/Task3_Thoracic_OAR'
	target_root = '/data/lbw/structseg2019/data_ori/crop_data_L-500W1800_wt'
	target_img_root = os.path.join(target_root, 'img')
	target_label_root = os.path.join(target_root, 'label')
	if not os.path.exists(target_root):
		os.mkdir(target_root)
	if not os.path.exists(target_img_root):
		os.mkdir(target_img_root)
	if not os.path.exists(target_label_root):
		os.mkdir(target_label_root)
	for dir in os.listdir(root):
		print(dir)
		img_path = '{}/{}/data.nii.gz'.format(root, dir)
		label_path = '{}/{}/label.nii.gz'.format(root, dir)
		target_img_dir = os.path.join(target_img_root, dir)
		target_label_dir = os.path.join(target_label_root, dir)
		if not os.path.exists(target_img_dir):
			os.mkdir(target_img_dir)
		if not os.path.exists(target_label_dir):
			os.mkdir(target_label_dir)

		img = read_nii(img_path, 1800, -500)
		label = load_nii(label_path).get_data().astype(np.uint8)

		if mode == 'gt':
			label_inds = np.where(label != 0)
			label_h, label_w, label_t = label_inds
			min_h, max_h = min(label_h), max(label_h)
			min_w, max_w = min(label_w), max(label_w)
			img_crop = img[min_h - 5: max_h + 5, min_w - 5: max_w + 5, :]
			label_crop = label[min_h - 5: max_h + 5, min_w - 5: max_w + 5, :]
		else:
			img_crop = img[100: 420, 140: 410, :]
			label_crop = label[100: 420, 140: 410, :]
		print(img_crop.shape)

		for i in range(img_crop.shape[0]):
			cv2.imwrite(os.path.join(target_img_dir, '{}.png'.format(i)), img_crop[i, :, :])
			cv2.imwrite(os.path.join(target_label_dir, '{}.png'.format(i)), label_crop[i, :, :])


def split_data():
	orien = 'wt'
	with open('./data_split/train_{}.txt'.format(orien), 'w') as f:
		print('new train_{}.txt'.format(orien))
	with open('./data_split/val_{}.txt'.format(orien), 'w') as f:
		print('new val_{}.txt'.format(orien))

	img_root = '/data/lbw/structseg2019/data_ori/crop_data_L-500W1800_{}/img'.format(orien)

	for i in range(1, 51):
		dir_path = os.path.join(img_root, str(i))
		img_num = len(os.listdir(dir_path))

		txt_path = './data_split/train_{}.txt'.format(orien) if i <= 40 else './data_split/val_{}.txt'.format(orien)
		for j in range(img_num):
			img_path = os.path.join(dir_path, '{}.png'.format(j))
			with open(txt_path, 'a') as f:
				f.write(img_path + '\r\n')


def generate_trachea_data():
	root = '/data/lbw/structseg2019/data_ori/Task3_Thoracic_OAR'
	# root = '/data/lbw/structseg2019/data_ori/re_labeling'
	target_root = '/data/ybh/PublicDataSet/StructSeg2019/single_class_data/esophagus_single'
	target_img_root = os.path.join(target_root, 'img')
	target_label_root = os.path.join(target_root, 'label')
	if not os.path.exists(target_root):
		os.mkdir(target_root)
	if not os.path.exists(target_img_root):
		os.mkdir(target_img_root)
	if not os.path.exists(target_label_root):
		os.mkdir(target_label_root)

	with open('./data_split/label_range.json', 'r') as f:
		label_range = json.load(f)

	for dir in os.listdir(root):
		print(dir)
		img_path = '{}/{}/data.nii.gz'.format(root, dir)
		label_path = '{}/{}/label.nii.gz'.format(root, dir)
		target_img_dir = os.path.join(target_img_root, dir)
		target_label_dir = os.path.join(target_label_root, dir)
		if not os.path.exists(target_img_dir):
			os.mkdir(target_img_dir)
		if not os.path.exists(target_label_dir):
			os.mkdir(target_label_dir)

		this_range = label_range[dir]

		img = read_nii(img_path, 324, 85)
		label = load_nii(label_path).get_data().astype(np.uint8)
		rl_h, rl_w, rl_t = np.where(label == 1)
		ll_h, ll_w, ll_t = np.where(label == 2)
		tr_h, tr_w, tr_t = np.where(label == 4)

		min_lung_h, max_lung_h = min(min(rl_h), min(ll_h)), max(max(rl_h), max(ll_h))
		min_lung_w, max_lung_w = min(min(rl_w), min(ll_w)), max(max(rl_w), max(ll_w))
		min_tr_t, max_tr_t = min(tr_t), max(tr_t)
		
		# img_crop = img[min_lung_h: max_lung_h, min_lung_w: max_lung_w, min_tr_t - 3: max_tr_t + 4]
		# label_crop = label[min_lung_h: max_lung_h, min_lung_w: max_lung_w, min_tr_t - 3: max_tr_t + 4]
		
		tr_h, tr_w, tr_t = np.where(label == 4)
		# ll_h, ll_w, ll_t = np.where(label == 2)

		min_trachea_t, max_trachea_t = min(min(tr_t), min(tr_t)), max(max(tr_t), max(tr_t))
		min_trachea_h, max_trachea_h = min(min(tr_h), min(tr_h)), max(max(tr_h), max(tr_h))
		min_trachea_w, max_trachea_w = min(min(tr_w), min(tr_w)), max(max(tr_w), max(tr_w))

		min_h, min_w, min_t, max_h, max_w, max_t = this_range['Trachea']

		img_crop = img[min_h - 10: max_h + 11, min_w - 10: max_w + 11, min_t - 5: max_t + 6]
		label_crop = label[min_h - 10: max_h + 11, min_w - 10: max_w + 11, min_t - 5: max_t + 6]
		print(img_crop.shape)
		
		for i in range(img_crop.shape[2]):
			cv2.imwrite(os.path.join(target_img_dir, '{}.png'.format(i)), img_crop[:, :, i])
			cv2.imwrite(os.path.join(target_label_dir, '{}.png'.format(i)), label_crop[:, :, i])


def generate_all_data():
	root = '/data/lbw/structseg2019/data_ori/Task3_Thoracic_OAR'
	target_root = '/data/ybh/PublicDataSet/StructSeg2019/single_class_data/all_organs'
	target_img_root = os.path.join(target_root, 'img')
	target_label_root = os.path.join(target_root, 'label')
	if not os.path.exists(target_root):
		os.mkdir(target_root)
	if not os.path.exists(target_img_root):
		os.mkdir(target_img_root)
	if not os.path.exists(target_label_root):
		os.mkdir(target_label_root)
	for dir in os.listdir(root):
		print(dir)
		img_path = '{}/{}/data.nii.gz'.format(root, dir)
		label_path = '{}/{}/label.nii.gz'.format(root, dir)
		target_img_dir = os.path.join(target_img_root, dir)
		target_label_dir = os.path.join(target_label_root, dir)
		if not os.path.exists(target_img_dir):
			os.mkdir(target_img_dir)
		if not os.path.exists(target_label_dir):
			os.mkdir(target_label_dir)
		
		img = read_nii(img_path, 1800, -500)
		label = load_nii(label_path).get_data().astype(np.uint8)
		# rl_h, rl_w, rl_t = np.where(label == 1)
		# ll_h, ll_w, ll_t = np.where(label == 2)
		# es_h, es_w, es_t = np.where(label == 5)
		# #
		# min_es_h, max_es_h, min_es_w, max_es_w, min_es_t, max_es_t = min(es_h), max(es_h), min(es_w), max(es_w), min(
		# 	es_t), max(es_t)
		# # # min_lung_h, max_lung_h = min(min(rl_h), min(ll_h)), max(max(rl_h), max(ll_h))
		# # # min_lung_w, max_lung_w = min(min(rl_w), min(ll_w)), max(max(rl_w), max(ll_w))
		# #
		# # img_crop = img[100: 420, 140: 410, min_es_t - 5: max_es_t + 6]
		# img_crop = img[min_es_h - 10: max_es_h + 11, min_es_w - 10: max_es_w + 11, min_es_t - 5: max_es_t + 6]
		# label_crop = label[min_es_h - 10: max_es_h + 11, min_es_w - 10: max_es_w + 11, min_es_t - 5: max_es_t + 6]
		# label_crop = label[100: 420, 140: 410, min_es_t - 5: max_es_t + 6]
		
		img_crop = img
		# label[label != 5] = 0
		# label[label == 5] = 1
		label_crop = label
		
		print(img_crop.shape)
		
		for i in range(img_crop.shape[2]):
			cv2.imwrite(os.path.join(target_img_dir, '{}.png'.format(i)), img_crop[:, :, i:i + 1])
			cv2.imwrite(os.path.join(target_label_dir, '{}.png'.format(i)), label_crop[:, :, i:i + 1])


def generate_esophagus_data():
	root = '/data/lbw/structseg2019/data_ori/Task3_Thoracic_OAR'
	target_root = '/data/ybh/PublicDataSet/StructSeg2019/single_class_data/trachea'
	target_img_root = os.path.join(target_root, 'img')
	target_label_root = os.path.join(target_root, 'label')
	if not os.path.exists(target_root):
		os.mkdir(target_root)
	if not os.path.exists(target_img_root):
		os.mkdir(target_img_root)
	if not os.path.exists(target_label_root):
		os.mkdir(target_label_root)
	for dir in os.listdir(root):
		print(dir)
		img_path = '{}/{}/data.nii.gz'.format(root, dir)
		label_path = '{}/{}/label.nii.gz'.format(root, dir)
		target_img_dir = os.path.join(target_img_root, dir)
		target_label_dir = os.path.join(target_label_root, dir)
		if not os.path.exists(target_img_dir):
			os.mkdir(target_img_dir)
		if not os.path.exists(target_label_dir):
			os.mkdir(target_label_dir)

		img = read_nii(img_path, 1180, -440)
		label = load_nii(label_path).get_data().astype(np.uint8)
		# rl_h, rl_w, rl_t = np.where(label == 1)
		# ll_h, ll_w, ll_t = np.where(label == 2)
		es_h, es_w, es_t = np.where(label == 5)
		#
		min_es_h, max_es_h, min_es_w, max_es_w, min_es_t, max_es_t = min(es_h), max(es_h), min(es_w), max(es_w), min(es_t), max(es_t)
		# # min_lung_h, max_lung_h = min(min(rl_h), min(ll_h)), max(max(rl_h), max(ll_h))
		# # min_lung_w, max_lung_w = min(min(rl_w), min(ll_w)), max(max(rl_w), max(ll_w))
		#
		# img_crop = img[100: 420, 140: 410, min_es_t - 5: max_es_t + 6]
		img_crop = img[min_es_h - 10: max_es_h + 11, min_es_w - 10: max_es_w + 11, min_es_t - 5: max_es_t + 6]
		label_crop = label[min_es_h - 10: max_es_h + 11, min_es_w - 10: max_es_w + 11, min_es_t - 5: max_es_t + 6]
		# label_crop = label[100: 420, 140: 410, min_es_t - 5: max_es_t + 6]
		
		# img_crop = img
		# label[label != 5] = 0
		# label[label == 5] = 1
		# label_crop = label
		
		print(img_crop.shape)

		for i in range(img_crop.shape[2]):
			cv2.imwrite(os.path.join(target_img_dir, '{}.png'.format(i)), img_crop[:, :, i:i + 1])
			cv2.imwrite(os.path.join(target_label_dir, '{}.png'.format(i)), label_crop[:, :, i:i + 1])


def generate_heart_data():
	root = '/data/lbw/structseg2019/data_ori/Task3_Thoracic_OAR'
	target_root = '/data/lbw/structseg2019/data_ori/heart_data'
	target_img_root = os.path.join(target_root, 'img')
	target_label_root = os.path.join(target_root, 'label')
	if not os.path.exists(target_root):
		os.mkdir(target_root)
	if not os.path.exists(target_img_root):
		os.mkdir(target_img_root)
	if not os.path.exists(target_label_root):
		os.mkdir(target_label_root)
	for dir in os.listdir(root):
		print(dir)
		img_path = '{}/{}/data.nii.gz'.format(root, dir)
		label_path = '{}/{}/label.nii.gz'.format(root, dir)
		target_img_dir = os.path.join(target_img_root, dir)
		target_label_dir = os.path.join(target_label_root, dir)
		if not os.path.exists(target_img_dir):
			os.mkdir(target_img_dir)
		if not os.path.exists(target_label_dir):
			os.mkdir(target_label_dir)

		img = read_nii(img_path, 350, 30)
		label = load_nii(label_path).get_data().astype(np.uint8)
		rl_h, rl_w, rl_t = np.where(label == 1)
		ll_h, ll_w, ll_t = np.where(label == 2)
		he_h, he_w, he_t = np.where(label == 3)

		min_he_t, max_he_t = min(he_t), max(he_t)
		min_lung_h, max_lung_h = min(min(rl_h), min(ll_h)), max(max(rl_h), max(ll_h))
		min_lung_w, max_lung_w = min(min(rl_w), min(ll_w)), max(max(rl_w), max(ll_w))

		img_crop = img[min_lung_h: max_lung_h, min_lung_w: max_lung_w, min_he_t - 3: max_he_t + 4]
		label_crop = label[min_lung_h: max_lung_h, min_lung_w: max_lung_w, min_he_t - 3: max_he_t + 4]
		print(img_crop.shape)

		for i in range(img_crop.shape[2]):
			cv2.imwrite(os.path.join(target_img_dir, '{}.png'.format(i)), img_crop[:, :, i:i + 1])
			cv2.imwrite(os.path.join(target_label_dir, '{}.png'.format(i)), label_crop[:, :, i:i + 1])


def generate_lung_data():
	root = '/data/lbw/structseg2019/data_ori/Task3_Thoracic_OAR'
	target_root = '/data/ybh/PublicDataSet/StructSeg2019/single_class_data/lung'
	target_img_root = os.path.join(target_root, 'img')
	target_label_root = os.path.join(target_root, 'label')
	if not os.path.exists(target_root):
		os.mkdir(target_root)
	if not os.path.exists(target_img_root):
		os.mkdir(target_img_root)
	if not os.path.exists(target_label_root):
		os.mkdir(target_label_root)
	for dir in os.listdir(root):
		print(dir)
		img_path = '{}/{}/data.nii.gz'.format(root, dir)
		label_path = '{}/{}/label.nii.gz'.format(root, dir)
		target_img_dir = os.path.join(target_img_root, dir)
		target_label_dir = os.path.join(target_label_root, dir)
		if not os.path.exists(target_img_dir):
			os.mkdir(target_img_dir)
		if not os.path.exists(target_label_dir):
			os.mkdir(target_label_dir)
		
		#700, -600
		img = read_nii(img_path, 1800, -500)
		label = load_nii(label_path).get_data().astype(np.uint8)
		rl_h, rl_w, rl_t = np.where(label == 1)
		ll_h, ll_w, ll_t = np.where(label == 2)

		min_lung_h, max_lung_h = min(min(rl_h), min(ll_h)), max(max(rl_h), max(ll_h))
		min_lung_w, max_lung_w = min(min(rl_w), min(ll_w)), max(max(rl_w), max(ll_w))
		min_lung_t, max_lung_t = min(min(rl_t), min(ll_t)), max(max(rl_t), max(ll_t))

		img_crop = img[min_lung_h - 10: max_lung_h + 11, min_lung_w - 10: max_lung_w + 11, min_lung_t - 5: max_lung_t + 6]
		label_crop = label[min_lung_h - 10: max_lung_h + 11, min_lung_w - 10: max_lung_w + 11, min_lung_t - 5: max_lung_t + 6]
		print(img_crop.shape)

		for i in range(img_crop.shape[2]):
			cv2.imwrite(os.path.join(target_img_dir, '{}.png'.format(i)), img_crop[:, :, i:i + 1])
			cv2.imwrite(os.path.join(target_label_dir, '{}.png'.format(i)), label_crop[:, :, i:i + 1])


def split_organ_data():
	organ = 'all_organs'
	with open('./2d_data_split/train_{}.txt'.format(organ), 'w') as f:
		print('new train_{}_InGT.txt'.format(organ))
	with open('./2d_data_split/val_{}.txt'.format(organ), 'w') as f:
		print('new val_{}_InGT.txt'.format(organ))

	img_root = '/data/ybh/PublicDataSet/StructSeg2019/single_class_data/{}/img'.format(organ)

	for i in range(1, 51):
		dir_path = os.path.join(img_root, str(i))
		img_num = len(os.listdir(dir_path))

		txt_path = './2d_data_split/train_{}.txt'.format(organ) if i <= 40 else './2d_data_split/val_{}.txt'.format(organ)
		for j in range(img_num):
			img_path = os.path.join(dir_path, '{}.png'.format(j))
			with open(txt_path, 'a') as f:
				f.write(img_path + '\r\n')


def split_heart_data():
	with open('./data_split/train_heart.txt', 'r') as f:
		all_train = [line[:-1] for line in f.readlines()]
	with open('./data_split/val_heart.txt', 'r') as f:
		all_val = [line[:-1] for line in f.readlines()]
	all_train.extend(all_val)

	for i in range(1, 6):
		for path in all_train:
			pid = int(path.split('/')[-2])
			if pid in range((i-1)*10+1, i*10+1):
				with open('./data_split/val_fold{}.txt'.format(i), 'a') as f:
					f.write(path + '\r\n')
			else:
				with open('./data_split/train_fold{}.txt'.format(i), 'a') as f:
					f.write(path + '\r\n')


def split_trachea_data():
	with open('./2d_data_split/train_trachea.txt', 'r') as f:
		all_train = [line[:-1] for line in f.readlines()]
	with open('./2d_data_split/val_trachea.txt', 'r') as f:
		all_val = [line[:-1] for line in f.readlines()]
	all_train.extend(all_val)

	for i in range(1, 6):
		for path in all_train:
			pid = int(path.split('/')[-2])
			if pid in range((i-1)*10+1, i*10+1):
				with open('./2d_data_split/trachea/val_fold{}.txt'.format(i), 'a') as f:
					f.write(path + '\r\n')
			else:
				with open('./2d_data_split/trachea/train_fold{}.txt'.format(i), 'a') as f:
					f.write(path + '\r\n')


def all_trachea_data():
	with open('./2d_data_split/train_trachea.txt', 'r') as f:
		all_train = [line[:-1] for line in f.readlines()]
	with open('./2d_data_split/val_trachea.txt', 'r') as f:
		all_val = [line[:-1] for line in f.readlines()]
	all_train.extend(all_val)
	
	for path in all_train:
		with open('./2d_data_split/trachea/all_trachea.txt', 'a') as f:
			f.write(path + '\r\n')


def split_lung_data():
	with open('./2d_data_split/train_lung.txt', 'r') as f:
		all_train = [line[:-1] for line in f.readlines()]
	with open('./2d_data_split/val_lung.txt', 'r') as f:
		all_val = [line[:-1] for line in f.readlines()]
	all_train.extend(all_val)

	for i in range(1, 6):
		for path in all_train:
			pid = int(path.split('/')[-2])
			if pid in range((i-1)*10+1, i*10+1):
				with open('./2d_data_split/lung/val_fold{}.txt'.format(i), 'a') as f:
					f.write(path + '\r\n')
			else:
				with open('./2d_data_split/lung/train_fold{}.txt'.format(i), 'a') as f:
					f.write(path + '\r\n')


def all_lung_data():
	with open('./2d_data_split/train_lung.txt', 'r') as f:
		all_train = [line[:-1] for line in f.readlines()]
	with open('./2d_data_split/val_lung.txt', 'r') as f:
		all_val = [line[:-1] for line in f.readlines()]
	all_train.extend(all_val)
	
	for path in all_train:
		with open('./2d_data_split/lung/all_lung.txt', 'a') as f:
			f.write(path + '\r\n')




def split_esophagus_data():
	with open('./2d_data_split/train_esophagus_single.txt', 'r') as f:
		all_train = [line[:-1] for line in f.readlines()]
	with open('./2d_data_split/val_esophagus_single.txt', 'r') as f:
		all_val = [line[:-1] for line in f.readlines()]
	all_train.extend(all_val)
	
	for i in range(1, 6):
		for path in all_train:
			pid = int(path.split('/')[-2])
			# if pid in range((i-1)*10+1, i*10+1):
			if pid in range(2*i-1, 2*i+1) or pid in range(8*i+3, 8*i+11):
				with open('./2d_data_split/esophagus_single_test/val_fold{}.txt'.format(i), 'a') as f:
					f.write(path + '\r\n')
			else:
				with open('./2d_data_split/esophagus_single_test/train_fold{}.txt'.format(i), 'a') as f:
					f.write(path + '\r\n')


def all_esophagus_data():
	with open('./2d_data_split/train_esophagus_single.txt', 'r') as f:
		all_train = [line[:-1] for line in f.readlines()]
	with open('./2d_data_split/val_esophagus_single.txt', 'r') as f:
		all_val = [line[:-1] for line in f.readlines()]
	all_train.extend(all_val)
	
	for path in all_train:
		with open('./2d_data_split/esophagus_single/all_esophagus.txt', 'a') as f:
			f.write(path + '\r\n')




def equal_spacing(arr, spacing, target, mode='trilinear'):
	"""

	:param arr: ndarray [h, w, t]
	:param spacing: tuple or list [3]
	:param target: float
	:return: arr_equal
	"""
	print('ori', arr.shape)
	target_h, target_w, target_t = int(arr.shape[0]*spacing[0]/target), int(arr.shape[1]*spacing[1]/target), int(arr.shape[2]*spacing[2]/target)
	arr = torch.from_numpy(arr[np.newaxis, np.newaxis, ...]).float()
	arr_equal = F.interpolate(arr, [target_h, target_w, target_t], mode=mode).numpy()[0, 0, ...].astype(np.uint8)
	print('after', arr_equal.shape)
	return arr_equal


def generate_vfn_pred():
	root = '/data/lbw/structseg2019/data/pred_ex059_val_fold1_npy_hard'
	target = '/data/lbw/structseg2019/data_ori/trachea_vfn_data/pred'
	with open('./data_split/img_spacing.json', 'r') as f:
		spacings = json.load(f)

	for name in os.listdir(root):
		print(name)
		path = os.path.join(root, name)
		pred = np.load(path)[0, 1, ...]
		ind = name[5:-4]
		spacing = spacings[ind]['spacing']
		pred = equal_spacing(pred, spacing, 1)

		target_path = os.path.join(target, name[:-4])
		np.save(target_path, pred)


def generate_vfn_img():
	root = '/data/lbw/structseg2019/data_ori/Task3_Thoracic_OAR'
	target = '/data/lbw/structseg2019/data_ori/trachea_vfn_data/img'
	with open('./data_split/img_spacing.json', 'r') as f:
		spacings = json.load(f)
	with open('./data_split/label_range.json', 'r') as f:
		label_range = json.load(f)

	for i in range(1, 51):
		print(i)
		img_path = '{}/{}/data.nii.gz'.format(root, i)
		img = read_nii(img_path, 85, 324)

		this_range = label_range[str(i)]
		# min_rl_h, min_rl_w, min_rl_t, max_rl_h, max_rl_w, max_rl_t = this_range['RightLung']
		# min_ll_h, min_ll_w, min_ll_t, max_ll_h, max_ll_w, max_ll_t = this_range['LeftLung']
		min_tr_h, min_tr_w, min_tr_t, max_tr_h, max_tr_w, max_tr_t = this_range['Trachea']
		min_h, min_w, min_t, max_h, max_w, max_t = min_tr_h-10, min_tr_w-10, min_tr_t-5, max_tr_h+11, max_tr_w+11, max_tr_t+6
		# min_h, min_w, max_h, max_w = min(min_rl_h, min_ll_h), min(min_rl_w, min_ll_w), max(max_rl_h, max_ll_h), max(
		#     max_rl_w, max_ll_w)
		# min_t, max_t = min_he_t - 3, max_he_t + 4

		img = img[min_h: max_h, min_w: max_w, min_t: max_t]

		spacing = spacings[str(i)]['spacing']
		img = equal_spacing(img, spacing, 1)

		target_path = os.path.join(target, 'img_{}'.format(i))
		np.save(target_path, img)


def generate_vfn_label():
	root = '/data/lbw/structseg2019/data_ori/Task3_Thoracic_OAR'
	# root = '/data/lbw/structseg2019/data_ori/re_labeling'
	target = '/data/lbw/structseg2019/data_ori/trachea_vfn_data/label'
	with open('./data_split/img_spacing.json', 'r') as f:
		spacings = json.load(f)
	with open('./data_split/label_range.json', 'r') as f:
		label_range = json.load(f)

	for i in range(1, 51):
	# for i in [3, 5, 28, 32]:
		print(i)
		label_path = '{}/{}/label.nii.gz'.format(root, i)
		label = np.uint8(load_nii(label_path).get_data())
		label[label!=4] = 0
		label[label==4] = 1

		this_range = label_range[str(i)]
		# min_rl_h, min_rl_w, min_rl_t, max_rl_h, max_rl_w, max_rl_t = this_range['RightLung']
		# min_ll_h, min_ll_w, min_ll_t, max_ll_h, max_ll_w, max_ll_t = this_range['LeftLung']
		min_tr_h, min_tr_w, min_tr_t, max_tr_h, max_tr_w, max_tr_t = this_range['Trachea']
		min_h, min_w, min_t, max_h, max_w, max_t = min_tr_h - 10, min_tr_w - 10, min_tr_t - 5, max_tr_h + 11, max_tr_w + 11, max_tr_t + 6
		# min_h, min_w, max_h, max_w = min(min_rl_h, min_ll_h), min(min_rl_w, min_ll_w), max(max_rl_h, max_ll_h), max(
		#     max_rl_w, max_ll_w)
		# min_t, max_t = min_he_t - 3, max_he_t + 4

		label = label[min_h: max_h, min_w: max_w, min_t: max_t]

		spacing = spacings[str(i)]['spacing']
		label = equal_spacing(label, spacing, 1, mode='nearest')

		target_path = os.path.join(target, 'label_{}'.format(i))
		np.save(target_path, label)


def split_vfn_data():
	with open('./data_split/train_vfn.txt', 'w') as f:
		pass
	with open('./data_split/val_vfn.txt', 'w') as f:
		pass

	for i in range(1, 51):
		target_txt = 'train' if i <= 40 else 'val'
		with open('./data_split/{}_vfn.txt'.format(target_txt), 'a') as f:
			f.write(str(i) + '\r\n')


if __name__ == '__main__':
	# data_distribution('/data/lbw/liver_data/labels_modified')
	# locate_zheyi_tumor('/data/lbw/liver_data/zheyi_label_nii', './data_split/zheyi_tumor_range.txt')
	# mha2nii('/data/lbw/zheyi_liver_mcl', '/data/lbw/liver_data/zheyi_label_nii')
	# max_range()
	# z_range()
	# generate_trachea_data()
	# generate_lung_data()
	# split_lung_data()
	# split_trachea_data()
	# generate_data('gt')
	# split_data()
	# generate_esophagus_data()
	# split_organ_data()
	# generate_all_data()
	# split_esophagus_data()
	all_trachea_data()
	# dump_range()
	# npy2nrrd()
	# split_heart_data()
	# dump_spacing()
	# generate_vfn_pred()
	# generate_vfn_img()
	# generate_vfn_label()
	# split_vfn_data()
	# split_trachea_data()
	print('done')

