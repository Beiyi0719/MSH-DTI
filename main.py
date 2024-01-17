# -*- coding: utf-8 -*-


import torch
import numpy as np
import torch.nn as nn
from model import GTN
from metric import get_metrics
from datalode import data_lode
import logging
import argparse
import warnings
import json
def main():

	device = 'cuda:0'
	warnings.filterwarnings("ignore", category=RuntimeWarning)
	parser = argparse.ArgumentParser()
	parser.add_argument('--epoch', default=1000, type=int)
	parser.add_argument('--kfold', default=10, type=int)
	parser.add_argument('--index', default=1, type=int)
	parser.add_argument('--layer', default=6, type=int)
	parser.add_argument('--lr', default=0.00001, type=float)
	parser.add_argument('--weight_decay', default=0.00001, type=float)
	parser.add_argument('--alpha', default=0.5, type=float)

	args = parser.parse_args()


	for i in range(10):
		print(args.weight_decay)
		max_auc = 0
		max_aupr = 0
		args.index = i + 1
		A, protein_structure, drug_structure, train_label, test_label, one_index, \
			zero_index = data_lode(args)

		drug_num = drug_structure.shape[0]
		protein_num = protein_structure.shape[0]
		model = GTN(drug_num=drug_num,
					protein_num=protein_num,
					A=A,
					drug_structure=drug_structure,
					protein_structure=protein_structure,
					args=args)

		class Myloss(nn.Module):
			def __init__(self):
				super(Myloss, self).__init__()

			def forward(self, iput, target):
				loss_sum = torch.pow((iput - target), 2)
				result = (1 - args.alpha) * ((target * loss_sum).sum()) + args.alpha * (((1 - target) * loss_sum).sum())
				return (result/2)
		myloss = Myloss()

		layers_params = list(map(id, model.parameters()))
		base_params = filter(lambda p: id(p) not in layers_params,
							 model.parameters())
		opt = torch.optim.Adam([{'params': base_params},
								{'params': model.parameters(), 'lr': 0.00001}]
							   , lr=args.lr, weight_decay=args.weight_decay)

		print(f'The {i + 1} fold')
		for i in range(args.epoch):
			for param in opt.param_groups:
				if param['lr'] > 0.001:
					param['lr'] *= 0.9
			# model = model
			model.train()
			opt.zero_grad()
			predict = model(A)
			print('-------------')
			loss = myloss(predict, train_label)
			print(f'epoch:  {i + 1}    train_loss:  {loss}')
			loss.backward(retain_graph=True)
			opt.step()

			with torch.no_grad():
				predict_test = predict.detach().cpu().numpy()
				predict_test_negative = predict_test[zero_index[0], zero_index[1]]
				predict_test_positive = predict_test[one_index[0], one_index[1]]

				predict_test_fold = np.concatenate((predict_test_positive, predict_test_negative))
				metrics = get_metrics(test_label, predict_test_fold)
				if i > 1:
					if max_auc < metrics[1] and max_aupr < metrics[0]:
						max_auc = metrics[1]

						max_aupr = metrics[0]
						data = {"i":i,"test": test_label.tolist(), "predict": predict_test_fold.tolist()}
						json_file_path = "data_layers_7.json"
						with open(json_file_path, "w") as json_file:
							json.dump(data, json_file)
					print(' test metrics:', metrics)

					print(f'AUPR: {max_aupr:.4f}  AUC:{max_auc:.4f}   ')
					# 配置日志输出的格式和级别
					logging.basicConfig(filename='result(2).log', level=logging.INFO,
										format='%(asctime)s - %(levelname)s - %(message)s')
					# 运行代码并将结果写入日志文件
					result = ('args.lr=',args.lr,'i=',i,' test metrics:', metrics,f'\nAUPR: {max_aupr:.4f}  AUC:{max_auc:.4f}')
					logging.info(f"运行结果: {result}")
if __name__ == '__main__':
	main()

