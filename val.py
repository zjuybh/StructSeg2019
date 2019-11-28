import argparse
from model.ssn import SSN
from model.ssn_cls import SSN_CLS
from model.unet import UNet
from model.denseunet import DenseUnet
from model.resunet import ResUNet
from model.se_resunet import SeResUNet, Shallow_SeResUNet
from model.deeplab_v3p import DeepLab
from model.resnet18 import Res18
from model.cbam_resunet import CbamResUNet
from model.dilated_resunet import DilatedResUnet
from model.vggunet import VGGUNet
import os
import torch
from imgaug import augmenters as iaa
from PIL import Image
from torchvision import transforms
import numpy as np
from utils import reverse_one_hot, get_label_info, colour_code_segmentation
import torch.nn.functional as F
import time
from utils import *
from dataset.Liver import Thoracic
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from torch.utils.data import DataLoader
import tqdm
import cv2
from nibabel import load as load_nii
import multiprocessing
import nibabel as nib
from skimage import measure
import json

# label_info = get_label_info('./class_dict.csv')
dices_record = []
name_record = []
all_organs = ['Bg', 'RightLung', 'LeftLung', 'Heart', 'Trachea', 'Esophagus', 'SpinalCord']


def postprocessing(pred):
	C = pred.shape[1]
	pred = np.argmax(pred.squeeze(), axis=0)
	post_pred = np.zeros(pred.shape, dtype=np.float)
	for i in range(1, C):
		pred_i = np.zeros(pred.shape, dtype=np.float)
		pred_i[pred != i] = 0
		pred_i[pred == i] = 1

		if i == 4:
			pred_res = pred_i
		else:
			[pred_res, num] = measure.label(pred_i, return_num=True)
			if num != 0:
				region = measure.regionprops(pred_res)
				box = []

				for j in range(num):
					box.append(region[j].area)
				label_num = box.index(max(box)) + 1
				pred_res[pred_res != label_num] = 0
				pred_res[pred_res == label_num] = 1
			else:
				pred_res = pred_i

		post_pred[pred_res == 1] = i

	post_pred = np.eye(C)[post_pred.astype(np.uint8)].transpose([3, 0, 1, 2])[np.newaxis, :].astype(np.float)
	return post_pred


def ensemble(roots, target_path='/data/lbw/liver_data/zheyi_pred_val_npy_ensemble', mode='avg'):
	if not os.path.exists(target_path):
		os.mkdir(target_path)
	names = os.listdir(roots[0])
	for name in names:
		print(name)
		path0 = os.path.join(roots[0], name)
		if mode == 'avg':
			predict = np.load(path0).astype(np.float) / 255
			for i, root in enumerate(roots):
				path = os.path.join(root, name)
				if i != 0:
					predict_cur = np.load(path).astype(np.float) / 255
					predict = predict + predict_cur
			predict = predict / len(roots)
			
			predict = (predict * 255).astype(np.uint8)
		else:
			predict = np.load(path0)
			predict = np.eye(7)[np.argmax(predict, axis=1)].transpose([0, 4, 1, 2, 3])
			for i, root in enumerate(roots):
				path = os.path.join(root, name)
				if i != 0:
					predict_cur = np.load(path)
					predict_cur = np.eye(7)[np.argmax(predict_cur, axis=1)].transpose([0, 4, 1, 2, 3])
					predict = predict + predict_cur
			predict = np.eye(7)[np.argmax(predict, axis=1)].transpose([0, 4, 1, 2, 3])
			predict = (predict * 255).astype(np.uint8)

		save_path = os.path.join(target_path, name[:-4])
		np.save(save_path, predict)


def merge(roots, cls_ind, target_path='/data/lbw/liver_data/zheyi_pred_val_npy_merge'):
	if not os.path.exists(target_path):
		os.mkdir(target_path)
	names = os.listdir(roots[0])

	with open('./data_split/label_range.json', 'r') as f:
		label_range = json.load(f)

	for name in names:
		print(name)
		this_range = label_range[name[5: -4]]
		predict = np.zeros([1, 7, 512, 512, this_range['shape'][2]], dtype=np.float)
		for i, root in enumerate(roots):
			path = os.path.join(root, name)
			predict_cur = np.load(path).astype(np.float) / 255
			for j in cls_ind[i]:
				predict[:, j: j+1] = predict_cur[:, j: j+1]
		predict = predict
		predict = (predict * 255).astype(np.uint8)

		save_path = os.path.join(target_path, name[:-4])
		np.save(save_path, predict)


def croped2full_pred(npy_root, cls, mode, loc=(100, 420, 140, 410), json_path='./data_split/loc.txt'):
	print('croped2full_pred...')
	npy_target = npy_root + '_full'
	if not os.path.exists(npy_target):
		os.mkdir(npy_target)

	with open('./data_split/label_range.json', 'r') as f:
		label_range = json.load(f)

	for name in os.listdir(npy_root):
		print(name)
		npy_path = os.path.join(npy_root, name)
		this_range = label_range[name[5: -4]]
		predict = np.zeros([1, 7, 512, 512, this_range['shape'][2]], dtype=np.uint8)
		predict_crop = np.load(npy_path)
		# predict_crop = thresh_pred(predict, 0.5, 0.5)
		
		if mode == 'fix':
			min_h, max_h, min_w, max_w = loc
			min_t = 0
			max_t = this_range['shape'][2]
		elif mode == 'lung':
			min_rl_h, min_rl_w, min_rl_t, max_rl_h, max_rl_w, max_rl_t = this_range[all_organs[1]]
			min_ll_h, min_ll_w, min_ll_t, max_ll_h, max_ll_w, max_ll_t = this_range[all_organs[2]]
			min_h, min_w, min_t, max_h, max_w, max_t = min(min_rl_h, min_ll_h), min(min_rl_w, min_ll_w), min(min_rl_t, min_ll_t), max(max_rl_h, max_ll_h), max(max_rl_w, max_ll_w), max(max_rl_t, max_ll_t)
			min_t -= 3
			max_t += 4
		elif mode == 'trachea':
			# min_rl_h, min_rl_w, min_rl_t, max_rl_h, max_rl_w, max_rl_t = this_range[all_organs[1]]
			# min_ll_h, min_ll_w, min_ll_t, max_ll_h, max_ll_w, max_ll_t = this_range[all_organs[2]]
			# min_h, min_w, min_t, max_h, max_w, max_t = min(min_rl_h, min_ll_h), min(min_rl_w, min_ll_w), min(min_rl_t, min_ll_t), max(max_rl_h, max_ll_h), max(max_rl_w, max_ll_w), max(max_rl_t, max_ll_t)
			# max_t += 10
			min_es_h, min_es_w, min_es_t, max_es_h, max_es_w, max_es_t = this_range[all_organs[5]]
			min_h, min_w, max_h, max_w = min_es_h - 10, min_es_w - 10, max_es_h + 11, max_es_w + 11
			min_t, max_t = min_es_t - 5, max_es_t + 6
			
		elif mode == 'tracheaInGT':
			min_tr_h, min_tr_w, min_tr_t, max_tr_h, max_tr_w, max_tr_t = this_range[all_organs[4]]
			min_h, min_w, min_t, max_h, max_w, max_t = min_tr_h-10, min_tr_w-10, min_tr_t-5, max_tr_h+11, max_tr_w+11, max_tr_t+6
		elif mode == 'esophagus':
			# min_rl_h, min_rl_w, min_rl_t, max_rl_h, max_rl_w, max_rl_t = this_range[all_organs[1]]
			# min_ll_h, min_ll_w, min_ll_t, max_ll_h, max_ll_w, max_ll_t = this_range[all_organs[2]]
			min_es_h, min_es_w, min_es_t, max_es_h, max_es_w, max_es_t = this_range[all_organs[4]]
			min_h, min_w, max_h, max_w = min_es_h - 10, min_es_w - 10, max_es_h + 11, max_es_w + 11
			min_t, max_t = min_es_t - 5, max_es_t + 6
		elif mode == 'heart':
			min_rl_h, min_rl_w, min_rl_t, max_rl_h, max_rl_w, max_rl_t = this_range[all_organs[1]]
			min_ll_h, min_ll_w, min_ll_t, max_ll_h, max_ll_w, max_ll_t = this_range[all_organs[2]]
			min_he_h, min_he_w, min_he_t, max_he_h, max_he_w, max_he_t = this_range[all_organs[3]]
			min_h, min_w, max_h, max_w = min(min_rl_h, min_ll_h), min(min_rl_w, min_ll_w), max(max_rl_h, max_ll_h), max(max_rl_w, max_ll_w)
			min_t, max_t = min_he_t - 3, max_he_t + 4
		elif mode == 'gt':
			min_h = min([this_range[all_organs[i]][0] for i in range(1, 7)]) - 5
			min_w = min([this_range[all_organs[i]][1] for i in range(1, 7)]) - 5
			max_h = max([this_range[all_organs[i]][3] for i in range(1, 7)]) + 5
			max_w = max([this_range[all_organs[i]][4] for i in range(1, 7)]) + 5
			min_t = 0
			max_t = this_range['shape'][2]
		else:
			min_h, max_h, min_w, max_w, min_t, max_t = this_range[all_organs[cls[-1]]]

		for ind, cls_ind in enumerate(cls):
			predict[:, cls_ind: cls_ind+1, min_h: max_h, min_w: max_w, min_t: max_t] = predict_crop[:, ind: ind+1]

		save_path = os.path.join(npy_target, name[:-4])
		np.save(save_path, predict)


def predict_on_image(model, dataloader, args):
	print('start val!')
	with torch.no_grad():
		model.eval()
		preds = dict()
		tq = tqdm.tqdm(total=len(dataloader) * args.batch_size)
		for i, (data, label_full, name) in enumerate(dataloader):
			if torch.cuda.is_available() and args.use_gpu:
				data = data.cuda()

			predict = model(data)

			predict = torch.nn.functional.interpolate(predict, label_full.shape[2:], mode='bilinear')

			predict = F.softmax(predict, dim=1).cpu().detach().numpy()

			for i in range(predict.shape[0]):
				predict_one = predict[i:i + 1, :, :, :].copy()

				name_pre = 'pred_' + name[i].split('/')[-2]
				if name_pre in preds:
					preds[name_pre].append(predict_one)
				else:
					preds[name_pre] = [predict_one]

			tq.update(args.batch_size)

		tq.close()

		for name in preds.keys():
			predict = np.stack(preds[name], axis=4)
			print(predict.shape)
			predict = (predict * 255).astype(np.uint8)

			save_path = os.path.join(args.save_npy_path, name)
			np.save(save_path, predict)

		print('val done!')


def eval_one(npy_path, name, save_nii, save_root, post_process):
	pred_path = os.path.join(npy_path, name)
	predict = np.load(pred_path).astype(np.float) / 255

	label_path = '/data/lbw/structseg2019/data_ori/Task3_Thoracic_OAR/{}/label.nii.gz'.format(name[5:-4])
	label = load_nii(label_path).get_data().astype(np.uint8)
	label = np.eye(7)[label].transpose([3, 0, 1, 2])[np.newaxis, :]

	if post_process:
		ori_shape = predict.shape
		predict = F.interpolate(torch.from_numpy(predict), [512, 512, 512], mode='trilinear')
		predict = F.interpolate(predict, ori_shape[2:], mode='trilinear').numpy()
		predict = postprocessing(predict)

	if save_nii:
		predict_nii = predict.copy()
		predict_nii = np.argmax(predict_nii.squeeze(), axis=0).astype(np.uint8)
		# predict_npy = predict_npy.transpose([1, 0, 2])
		predict_nii = nib.Nifti1Image(predict_nii, None)
		nib.save(predict_nii, os.path.join(save_root, name[:-4]))

	dices = compute_multi_dice(predict, label)
	print(name[:-4])
	for i in range(7):
		print(all_organs[i], float(dices[i]))

	return [dices, '{0:02d}'.format(int(name[5:-4]))]


def record_dice(dices):
	global dices_record, name_record
	dices_record.append(dices[0])
	name_record.append(dices[1])


def evaluate(npy_path, csv_path, save_nii=False, save_root='/data/lbw/liver_data/preds_val', n_process=4, post_process=False):
	global dices_record, name_record
	if not os.path.exists(save_root):
		os.mkdir(save_root)

	pool = multiprocessing.Pool(processes=n_process)

	for name in os.listdir(npy_path):
		pool.apply_async(eval_one, (npy_path, name, save_nii, save_root, post_process), callback=record_dice)

	pool.close()
	pool.join()

	mean_dices = []
	W = [0, 100, 100, 100, 70, 80, 100]
	for dices in dices_record:
		mean_dices.append(sum([a * b / sum(W) for a, b in zip(W, dices)]))
	mean_dice = np.mean(mean_dices)

	dices_record = np.array(dices_record)
	overall = ['pred_mean']
	print('average:')
	for i in range(7):
		organs_i = np.mean(dices_record, axis=0)[i]
		overall.append(organs_i)
		print(all_organs[i], organs_i)
	overall.append(mean_dice)
	print('mean', mean_dice)

	# rank
	name_record = np.array(name_record).reshape([-1, 1])
	mean_dices = np.array(mean_dices).reshape([-1, 1])
	data = np.concatenate([name_record, dices_record, mean_dices], axis=1)
	data = np.concatenate([data, np.array(overall).reshape(1, -1)], axis=0)
	columns = ['Name', 'Bg', 'RightLung', 'LeftLung', 'Heart', 'Trachea', 'Esophagus', 'SpinalCord', 'Mean']
	result = pd.DataFrame(data, columns=columns).sort_values('Name')
	result_hard = pd.DataFrame(data, columns=columns).sort_values('Mean').values[:, 0]
	for name in result_hard:
		if name == 'pred_mean':
			break
		with open('./data_split/hard.txt', 'a') as f:
			f.write(name + '\r\n')

	result.to_csv(csv_path, index=False)
	print('done')


def main(params):
	# basic parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default="lits", help='Dataset you are using.')
	parser.add_argument('--fold', type=str, default='1', help='predict on which fold')
	parser.add_argument('--video', action='store_true', default=False, help='predict on video')
	parser.add_argument('--in_range', type=bool, default=False, help='Whether use adjacent img as input.')
	parser.add_argument('--dropout', type=int, default=0, help='Whether use dropout.')
	parser.add_argument('--seq', type=int, default=0, help='Whether use adjacent img as input.')
	parser.add_argument('--model', type=str, default="unet", help='Model you are using.')
	parser.add_argument('--num_workers', type=int, default=8, help='num of workers')
	parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
	parser.add_argument('--checkpoint_path', type=str, default=None, help='The path to the pretrained weights of model')
	parser.add_argument('--deep_supervision', type=int, default=0, help='length of input images')
	parser.add_argument('--context_path', type=str, default="resnet34", help='The context path model you are using.')
	parser.add_argument('--num_classes', type=int, default=4, help='num of object classes (with void)')
	parser.add_argument('--data', type=str, default=None, help='Path to image or video for prediction')
	parser.add_argument('--crop_h', type=int, default=448, help='Height of cropped/resized input image to network')
	parser.add_argument('--crop_w', type=int, default=448, help='Width of cropped/resized input image to network')
	parser.add_argument('--val_path', type=str, default="./data_split/lits_val.txt", help='Path of val list.')
	parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
	parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
	parser.add_argument('--csv_path', type=str, default=None, required=True, help='Path to label info csv file')
	parser.add_argument('--save_pred', type=bool, default=False, help='Whether to save predict image')
	parser.add_argument('--save_npy_path', type=str, default=None, required=True, help='Path to save predict image')
	parser.add_argument('--len_seq', type=int, default=3, help='Whether to use adjacent slice for input')

	args = parser.parse_args(params)

	print('save_pred', args.save_pred)
	print('in_liver', args.in_range)
	seq = True if args.seq != 0 else False
	print('img_seq', seq)
	dropout = True if args.dropout != 0 else False
	print('dropout', dropout)
	dataset_val = Thoracic(args.val_path, (args.crop_h, args.crop_w), organs, mode='val', aug=False,
						   img_seq=seq, len_seq=args.len_seq, tta=False)

	print('size of valset', len(dataset_val))
	dataloader_val = DataLoader(
		dataset_val,
		# this has to be 1
		batch_size=args.batch_size,
		shuffle=False,
		num_workers=args.num_workers
	)

	# build model
	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

	in_channel = args.len_seq
	
	ds_flag = (args.deep_supervision == 1)
	print('loading model...')
	if args.model == 'ssn':
		model = SSN(args.num_classes, args.context_path, in_channel=in_channel).cuda()
	elif args.model == 'denseunet':
		model = DenseUnet(in_channel, args.num_classes).cuda()
	elif args.model == 'unet':
		model = UNet(in_channel, args.num_classes, deep_supervision=ds_flag).cuda()
	
	elif args.model == 'resunet':
		model = ResUNet(in_channel, args.num_classes, deep_supervision=ds_flag).cuda()
	elif args.model == 'se_resunet':
		model = SeResUNet(in_channel, args.num_classes, deep_supervision=ds_flag, dropout=dropout, rate=0.1).cuda()
		# model = Shallow_SeResUNet(in_channel, args.num_classes, deep_supervision=ds_flag, dropout=dropout,
		# 						  rate=0.1).cuda()

	elif 'vgg' in args.model:
		encoder = args.model[:-5]
		model = VGGUNet(in_channel, args.num_classes, encoder=encoder, pretrain=True, deep_supervision=ds_flag,
						dropout=dropout, rate=0.1).cuda()

	elif args.model == 'deeplabv3p':
		model = DeepLab(args.num_classes).cuda()
	else:
		print('NOT VALID MODEL NAME !!!')

	if torch.cuda.is_available() and args.use_gpu:
		model = torch.nn.DataParallel(model).cuda()

	# load pretrained model if exists
	print('load model from %s ...' % args.checkpoint_path)
	model.module.load_state_dict(torch.load(args.checkpoint_path))
	print('Done!')

	if not os.path.exists(args.save_npy_path):
		os.mkdir(args.save_npy_path)
	
	predict_on_image(model, dataloader_val, args)
	
	# evaluate(args)
	
	
if __name__ == '__main__':
	# organs = ['Bg', 'RightLung', 'LeftLung', 'Heart', 'Esophagus', 'Trachea', 'SpinalCord']
	organs = ['Bg', 'Trachea']
	val_set = 'val_fold2'
	n_process = 10
	ckpt_name = 'ex13_se_resunet_bs16_h128_w128_epo5000_dice_sgd_esophagus_dropout0_ex18_config_fold2_epo165iter23760.pth'
	ckpt_path = './checkpoints_final/{}/{}'.format('_'.join(ckpt_name.split('_')[:-1]), ckpt_name)
	save_npy_path = '/data/ybh/Programs/StructSeg2019_2d/data/pred_{}_{}_npy'.format(ckpt_name.split('_')[0], val_set)
	save_nii_path = save_npy_path.replace('npy', 'nii')
	save_csv_path = './results/{}_epo{}_{}.csv'.format(ckpt_name.split('_')[0], ckpt_name.split('epo')[-1][:-4], val_set)
	params = [
		'--dataset', 'Thoracic',
		'--val_path', './2d_data_split/{}.txt'.format(val_set),
		'--model', 'se_resunet',
		# '--model', 'vgg19_unet',
		
		'--context_path', 'resnet101',
		'--crop_h', '128',
		'--crop_w', '128',
		'--num_classes', str(len(organs)),
		'--batch_size', '1',
		'--dropout', '0',
		'--seq', '0',
		'--len_seq', '3',
		'--deep_supervision', '1',
		'--csv_path', './class_dict.csv',
		'--checkpoint_path', ckpt_path,
		'--cuda', '5',
		# '--save_pred', 'True',
		# '--in_liver', 'True',
		'--save_npy_path', save_npy_path,
		# '--img_seq', 'True',
	]
	#
	main(params)
	croped2full_pred(save_npy_path, [0, 4], 'esophagus', loc=(0, 512, 0, 512), json_path='./data_split/loc.txt')
	evaluate(save_npy_path + '_full', save_csv_path, save_nii=True, save_root=save_nii_path, n_process=n_process,
			 post_process=False)
	
	# merge(['/data/lbw/structseg2019/data/pred_ex011_val_npy_full', '/data/lbw/structseg2019/data/pred_ex034_val_npy_full',
	#           '/data/lbw/structseg2019/data/pred_ex031_val_npy_full', '/data/lbw/structseg2019/data/pred_ex016_val_npy_full',
	#           '/data/lbw/structseg2019/data/pred_ex033_val_npy_full'], [[0, 1, 2], [3], [4], [5], [6]], target_path='/data/lbw/structseg2019/data/zheyi_pred_val_npy_merge')
	#
	# exs = [52, 61, 63]
	# target_path = '/data/ybh/Programs/StructSeg2019_2d/data/pred_ensemble_{}_val_npy'.format('_'.join([str(ex) for ex in exs]))
	# ensemble_csv_path = './results/ex_{}_val.csv'.format('_'.join([str(ex) for ex in exs]))
	# ensemble_nii_path = '/data/ybh/Programs/StructSeg2019_2d/data/pred_ensemble_{}_val_nii'.format('_'.join([str(ex) for ex in exs]))
	# ensemble(['/data/ybh/Programs/StructSeg2019_2d/data/pred_ex{:0>2d}_val_trachea_npy_full'.format(ex) for ex in exs],
	#          target_path=target_path, mode='avg')
	#
	# # target_path = '/data/lbw/structseg2019/data/pred_val_npy_merge'
	# # ensemble_csv_path = './results/merge_val_postprocessing.csv'
	# # ensemble_nii_path = '/data/lbw/structseg2019/data/pred_val_nii_merge2'
	# evaluate(target_path, ensemble_csv_path, save_nii=True, save_root=ensemble_nii_path, n_process=n_process,
	#          post_process=False)
	