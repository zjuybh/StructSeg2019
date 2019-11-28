import argparse
from torch.utils.data import DataLoader
from dataset.Liver import Thoracic
import os
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
from model.vgg19unet import VGG19UNet
from model.r2unet import *
from model.factory import AlbuNet
import torch
import torch.nn
import torch.nn.init
from tensorboardX import SummaryWriter
import tqdm
from torch.nn import functional as F
import numpy as np
from utils import reverse_one_hot, get_label_info, colour_code_segmentation, compute_global_accuracy
from utils import *
import random


weights = {'Bg': 1,
		  'RightLung': 1,
		  'LeftLung': 1,
		  'Heart': 1,
		  'Esophagus': 15,
		  'Trachea': 5,
		  'SpinalCord': 10}

eval_weight = {'Bg': 0,
		  'RightLung': 100,
		  'LeftLung': 100,
		  'Heart': 100,
		  'Esophagus': 70,
		  'Trachea': 80,
		  'SpinalCord': 100}

dice_weight = {'Bg': 1,
		  'RightLung': 1,
		  'LeftLung': 1,
		  'Heart': 1,
		  'Esophagus': 1,
		  'Trachea': 1,
		  'SpinalCord': 1}


def val(args, model, dataloader, loss_func):
	print('start val!')
	with torch.no_grad():
		model.eval()
		dice_record = []
		mean_dice_record = []
		loss_record = []

		if args.loss == 'wce':
			W = [weights[organ] for organ in organs]
		# organs = ['Bg', 'RightLung', 'LeftLung', 'Heart', 'Trachea', 'Esophagus', 'SpinalCord']
		elif args.loss == 'dice':
			W = [dice_weight[organ] for organ in organs]
		else:
			W = None
		preds = dict()
		labels = dict()
		cnt_cls = np.zeros([1, len(organs) - 1], dtype=np.float)
		for i, (data, label_full, name) in enumerate(dataloader):

			data = data.cuda()
			label_full = label_full.cuda()

			predict = model(data)

			predict = F.interpolate(predict, label_full.shape[2:], mode='bilinear')

			loss = loss_func(label_full, F.softmax(predict, dim=1), W=W)
			loss_record.append(loss.item())

			predict = F.softmax(predict, dim=1).cpu().detach().numpy()

			label_full = label_full.cpu().detach().numpy()

			for i in range(predict.shape[0]):
				suffix = len(name[i].split('/')[-1])
				name_pre = name[i][:-suffix]
				if name_pre in preds:
					preds[name_pre].append(predict[i:i + 1, :, :, :])
					labels[name_pre].append(label_full[i:i + 1, :, :, :])
				else:
					preds[name_pre] = [predict[i:i + 1, :, :, :]]
					labels[name_pre] = [label_full[i:i + 1, :, :, :]]

		for name in preds.keys():
			predict = np.stack(preds[name], axis=2)
			label = np.stack(labels[name], axis=2)

			dices = compute_multi_dice(predict, label)
			# W = [0, 100, 100, 100, 80, 70, 100]
			eval_W = [eval_weight[organ] for organ in organs]
			mean_dice_record.append(sum([a * b / sum(eval_W) for a, b in zip(eval_W, dices)]))
			# mean_dice_record.append(np.mean([dices]))
			dice_record.append(dices)

		mean_dice = np.mean(mean_dice_record)
		mean_loss = np.mean(loss_record)
		dice = np.array(dice_record).mean(axis=0)

		cls_acc = cnt_cls / (len(dataloader))

		for i in range(len(organs)):
			print('{:11} dice: {:.5f}'.format(organs[i], float(dice[i])))
		print('mean dice', float(mean_dice))

		return mean_dice, dice, mean_loss


def train(args, model, optimizer, scheduler, loss_func, dataloader_train, dataloader_val, ckpt_path):
	from datetime import datetime
	current_time = datetime.now().strftime('%b%d_%H-%M-%S')
	writer = SummaryWriter(log_dir=args.log_dir + '_' + current_time)
	step = 0
	val_cnt = 0
	best_dice = -1
	n_func = len(loss_func)
	if n_func == 1:
		loss_func = loss_func[0]
	else:
		loss_func, loss_func2 = loss_func

	len_loader = len(dataloader_train)

	tq = tqdm.tqdm(total=len_loader * args.batch_size)
	tq.set_description('epoch %d, lr %f' % (0, args.learning_rate))
	model.train()
	for epoch in range(args.num_epochs):
		# scheduler.step()
		loss_record = []
		for i, (data, label) in enumerate(dataloader_train):
			
			data = data.cuda()
			# cls_label = cls_label.cuda()
			label = label.cuda()
			
			if args.loss == 'wce':
				W = [weights[organ] for organ in organs]
				# W = [1, 1, 1, 1, 10, 5, 3]
			# elif args.loss == 'dice':
			#     W = [1, 1, 1, 1, 1, 1, 1]
			else:
				W = None
			
			if args.model == 'ssn':
				output, output_sup1, output_sup2 = model(data)
				loss1 = loss_func(label, F.softmax(output, dim=1), W=W)
				loss2 = loss_func(label, F.softmax(output_sup1, dim=1), W=W)
				loss3 = loss_func(label, F.softmax(output_sup2, dim=1), W=W)
				loss = loss1 + loss2 + loss3
				if n_func != 1:
					loss4 = loss_func2(label, F.softmax(output, dim=1), W=W)
					loss5 = loss_func2(label, F.softmax(output_sup1, dim=1), W=W)
					loss6 = loss_func2(label, F.softmax(output_sup2, dim=1), W=W)
					loss = loss + 0.1 * (loss4 + loss5 + loss6)
			
			elif args.deep_supervision == 1:
				y0, y1, y2, y3, y4 = model(data)
				loss0 = loss_func(label, F.softmax(y0, dim=1), W=W)
				loss1 = loss_func(label, F.softmax(y1, dim=1), W=W)
				loss2 = loss_func(label, F.softmax(y2, dim=1), W=W)
				loss3 = loss_func(label, F.softmax(y3, dim=1), W=W)
				loss4 = loss_func(label, F.softmax(y4, dim=1), W=W)
				loss = loss0 + loss1 + loss2 + loss3 + loss4
			
			# elif args.model == 'ssncls':
			#     cls_pred, output, output_sup1, output_sup2 = model(data)
			#     loss_cls = torch.nn.BCEWithLogitsLoss()(cls_pred, cls_label)
			#     loss1 = loss_func(label, F.softmax(output, dim=1), W=W)
			#     loss2 = loss_func(label, F.softmax(output_sup1, dim=1), W=W)
			#     loss3 = loss_func(label, F.softmax(output_sup2, dim=1), W=W)
			#     loss = loss_cls + loss1 + loss2 + loss3
			#     if n_func != 1:
			#         loss4 = loss_func2(label, F.softmax(output, dim=1), W=W)
			#         loss5 = loss_func2(label, F.softmax(output_sup1, dim=1), W=W)
			#         loss6 = loss_func2(label, F.softmax(output_sup2, dim=1), W=W)
			#         loss = loss + 0.1 * (loss4 + loss5 + loss6)
			
			else:
				output = model(data)
				loss = loss_func(label, F.softmax(output, dim=1), W=W)
				if n_func != 1:
					loss2 = loss_func2(label, F.softmax(output, dim=1), W=W)
					loss = loss + 0.1 * (loss2)
			
			tq.update(args.batch_size)
			tq.set_postfix(loss='%.6f' % loss)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			step += 1
			writer.add_scalar('train/loss_step', loss, step)
			loss_record.append(loss.item())
			
			if i == len(dataloader_train) - 1:
			# if i == 0:
				tq.close()

				val_cnt += 1

				loss_train_mean = np.mean(loss_record)

				scheduler.step(loss_train_mean)

				writer.add_scalar('train/loss_train_perval', float(loss_train_mean), val_cnt)
				print('loss for train : %f' % loss_train_mean)
				loss_record = []

				# if use all data to train, not have to val
				mean_dice, dice, val_mean_loss = val(args, model, dataloader_val, loss_func=loss_func)
				writer.add_scalar('val/' + 'mean_dice' + '_val', float(mean_dice), val_cnt)
				writer.add_scalar('val/' + 'val_mean_loss' + '_val', float(val_mean_loss), val_cnt)

				# scheduler.step(val_mean_loss)

				for i in range(len(organs)):
					writer.add_scalar('val/' + organs[i] + '_val', float(dice[i]), val_cnt)

				if mean_dice > best_dice:
					best_dice = mean_dice
					if not os.path.isdir(args.save_model_path):
						os.mkdir(args.save_model_path)
					save_name = '_'.join([args.log_dir.split('/')[-1], 'epo%diter%d.pth' % (epoch + 1, step)])
					torch.save(model.module.state_dict(), os.path.join(ckpt_path, save_name))
				elif (epoch+1) % 10 == 0:
				# if (epoch+1) % 10 == 0:
					save_name = '_'.join([args.log_dir.split('/')[-1], 'epo%d.pth' % (epoch+1)])
					torch.save(model.module.state_dict(), os.path.join(ckpt_path, save_name))

				tq = tqdm.tqdm(total=len_loader * args.batch_size)
				tq.set_description('epoch %d, lr %f' % (epoch + 1, optimizer.param_groups[0]['lr']))

				model.train()


def main(params):
	# basic parameters
	parser = argparse.ArgumentParser()
	parser.add_argument('--log_dir', type=str, help='name of tensorboard lod dir')
	parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
	parser.add_argument('--model', type=str, default="unet", help='Model you are using.')
	parser.add_argument('--optimizer', type=str, default="sgd", help='Optimizer you are using.')
	parser.add_argument('--loss', type=str, default="ce", help='Loss function you are using.')
	parser.add_argument('--deep_supervision', type=int, default=0, help='length of input images')
	parser.add_argument('--dataset', type=str, default="lits", help='Dataset you are using.')
	# parser.add_argument('--csv_path', type=str, default="./class_dict.csv", help='Dataset you are using.')
	parser.add_argument('--seq', type=int, default=0, help='Whether use adjacent img as input.')
	parser.add_argument('--len_seq', type=int, default=1, help='length of input images')
	parser.add_argument('--dropout', type=int, default=0, help='Whether use dropout.')
	parser.add_argument('--train_path', type=str, default="./data_split/train.txt", help='Path of training list.')
	parser.add_argument('--val_path', type=str, default="./data_split/val.txt", help='Path of val list.')
	parser.add_argument('--test_path', type=str, default="./data_split/lits_test.txt", help='Path of test list.')
	parser.add_argument('--crop_h', type=int, default=512, help='Height of cropped/resized input image to network')
	parser.add_argument('--crop_w', type=int, default=512, help='Width of cropped/resized input image to network')
	parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
	parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using.')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate used for train')
	parser.add_argument('--num_workers', type=int, default=8, help='num of workers')
	parser.add_argument('--num_classes', type=int, default=7, help='num of object classes (with void)')
	parser.add_argument('--cuda', type=str, default='2', help='GPU ids used for training')
	parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
	parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
	
	
	args = parser.parse_args(params)
	
	print(args)
	
	def seed_torch(seed=0):
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)

	# set seed for torch and numpy
	seed_torch(0)

	dropout = True if args.dropout != 0 else False
	print('dropout', dropout)
	dataset_train = Thoracic(args.train_path, (args.crop_h, args.crop_w), organs, mode='train', aug=True,
	                         img_seq=args.seq, len_seq=args.len_seq, hard=False, pred=False)

	print('size of trainset', len(dataset_train))
	dataloader_train = DataLoader(
		dataset_train,
		batch_size=args.batch_size,
		shuffle=True,
		num_workers=args.num_workers
	)

	dataset_val = Thoracic(args.val_path, (args.crop_h, args.crop_w), organs, mode='val', aug=False,
	                       img_seq=args.seq, len_seq=args.len_seq, hard=False, pred=False)

	print('size of valset', len(dataset_val))
	dataloader_val = DataLoader(
		dataset_val,
		# this has to be 1
		batch_size=1,
		shuffle=False,
		num_workers=args.num_workers
	)

	# build model
	def weights_init(m):
		classname = m.__class__.__name__
		if (classname.find('Conv') != -1) and ('Block' not in classname) and ('Relu' not in classname):
			torch.nn.init.xavier_uniform_(m.weight.data)
			if m.bias is not None:
				torch.nn.init.constant_(m.bias.data, 0.0)

	os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
	
	if args.seq == 1:
		in_channel = int(args.len_seq)
	else:
		in_channel = 3
	
	print('in channel', in_channel)
	
	print('loading model...')
	ds_flag = (args.deep_supervision == 1)
	
	if args.model == 'ssn':
		model = SSN(args.num_classes, args.context_path, in_channel=in_channel).cuda()
	elif args.model == 'denseunet':
		model = DenseUnet(in_channel, args.num_classes).cuda()
	elif args.model == 'unet':
		model = UNet(in_channel, args.num_classes, deep_supervision=ds_flag).cuda()
	
	elif args.model == 'r2attunet':
		model = R2AttU_Net(in_channel, args.num_classes).cuda()
	elif args.model == 'resunet':
		model = ResUNet(in_channel, args.num_classes, deep_supervision=ds_flag).cuda()
	elif args.model == 'se_resunet':
		model = SeResUNet(in_channel, args.num_classes, deep_supervision=ds_flag, dropout=dropout, rate=0.1).cuda()
		# model = Shallow_SeResUNet(in_channel, args.num_classes, deep_supervision=ds_flag, dropout=dropout, rate=0.1).cuda()
		
	elif 'vgg' in args.model:
		encoder = args.model[:-5]
		model = VGGUNet(in_channel, args.num_classes, encoder = encoder, pretrain=True, deep_supervision=ds_flag, dropout = dropout, rate=0.1).cuda()
	
	elif args.model == 'deeplabv3p':
		model = DeepLab(args.num_classes).cuda()
	else:
		print('NOT VALID MODEL NAME !!!')
	
	if args.model in ['unet', 'resunet', 'se_resunet', 'cbam_resunet', 'dilated_resunet', 'unetres34']:
		print('initializing model weights...')
		model.apply(weights_init)
	if args.model in ['r2attunet']:
		init_weights(model)
	
	model = torch.nn.DataParallel(model).cuda()
	print('load successfully!')
	
	# load pretrained model if exists
	if args.pretrained_model_path is not None:
		print('load model from %s ...' % args.pretrained_model_path)
		model.module.load_state_dict(torch.load(args.pretrained_model_path))
		model.module.outc = nn.Conv2d(64, len(organs), 1).cuda()
		print('Done!')
	
	# build optimizer
	# optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
	if args.optimizer == 'sgd':
		#weight_decay=5e-4
		optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
	else:
		optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
	
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, verbose=True)
	# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120, 160], gamma=0.3)
	
	# focal loss or ce loss
	if args.loss == 'ce':
		loss_func = [XentLoss(bWeighted=False, gamma=0, bMask=False)]
	elif args.loss == 'focal':
		loss_func = [XentLoss(bWeighted=False, gamma=2, bMask=False)]
	elif args.loss == 'wce':
		loss_func = [XentLoss(bWeighted=True, gamma=0, bMask=False)]
	elif args.loss == 'dice':
		loss_func = [MulticlassDiceLoss()]
	elif args.loss == 'ce_dice':
		loss_func = [XentLoss(bWeighted=False, gamma=0, bMask=False), MulticlassDiceLoss()]
	
	ckpt_path = os.path.join(args.save_model_path, args.log_dir[11:])
	if not os.path.exists(ckpt_path):
		os.mkdir(ckpt_path)
	
	# train
	train(args, model, optimizer, scheduler, loss_func, dataloader_train, dataloader_val, ckpt_path)
	
	
	
if __name__ == '__main__':
	ex = '76'
	# organs = ['Bg', 'RightLung', 'LeftLung', 'Heart', 'Esophagus', 'Trachea', 'SpinalCord']
	# organs = ['Bg', 'RightLung', 'LeftLung']
	organs = ['Bg', 'Trachea']
	model = 'se_resunet'
	seq = '1'
	len_seq = '3'
	
	batch_size = '12'
	crop_h = '128'
	crop_w = '128'
	cuda = '3'
	num_epochs = '500'
	loss = 'wce'
	optimizer = 'sgd'
	dataset = 'trachea'
	# dataset = 'All'
	dropout = '1'
	# ps = 'all_lung_common_aug_Z_score_ds'
	# ps = 'Esophagus_single_aug_shear0.10_ds_seed2'
	# ps = 'ex18_config_all_esophagus'
	ps = 'ex61_config_all_trachea'
	# ps = 'Trachea_aug_ds_seq3_weight1_3'
	# ps = 'All_aug'
	deep_supervision = '1'
	
	params = [
		'--dataset', dataset,
		# '--train_path', './2d_data_split/train_lung.txt',
		# '--val_path', './2d_data_split/val_lung.txt',
		
		'--train_path', './2d_data_split/trachea/all_trachea.txt',
		'--val_path', './2d_data_split/trachea/all_trachea.txt',
		# '--pretrained_model_path', ckpt_path,
		'--num_classes', str(len(organs)),
		'--learning_rate', '0.01',
		'--num_workers', '8',
		'--save_model_path', './checkpoints_final',
		'--deep_supervision', deep_supervision,
		'--dropout', dropout,
		'--cuda', cuda,
		'--num_epochs', num_epochs,
		'--model', model,
		'--optimizer', optimizer,
		'--loss', loss,
		'--crop_h', crop_h,
		'--crop_w', crop_w,
		
		'--seq', seq,
		'--len_seq', len_seq,
		
		'--batch_size', batch_size,
		'--log_dir',
		'runs_final/ex{}_{}_bs{}_h{}_w{}_epo{}_{}_{}_{}_dropout{}_{}'.format(ex, model, batch_size, crop_h, crop_w, num_epochs, loss, optimizer,
														   dataset, dropout, ps),
	]
	main(params)
	print('done')
	input()



