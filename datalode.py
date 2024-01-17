# -*- coding: utf-8 -*-

import numpy as np
import torch
from process_sim import create_sim

def data_lode(args):
	drug_drug_path = 'data/mat_drug_drug.txt'
	protein_protein_path = 'data/mat_protein_protein.txt'

	drug_drug = np.loadtxt(drug_drug_path)
	drug_sim = create_sim()
	protein_protein = np.loadtxt(protein_protein_path)

	drug_protein = np.load('data/drug_protein.npy')
	pro_str = torch.load("pro_fea.pt")
	dru_str = torch.load("drug_fea.pt")


	drug_num = drug_protein.shape[0]
	protein_num = drug_protein.shape[1]
	num = drug_num + protein_num
	
	pre_train_data = np.copy(drug_protein)
	np.random.seed(10)
	one_index = np.where(drug_protein == 1)
	# 堆叠，变成坐标形式
	one_index = np.stack(one_index).T
	np.random.shuffle(one_index)
	one_index = one_index.T
	one_num = int((one_index.shape[1])/args.kfold)
	# 十折交叉验证
	if args.index==args.kfold:
		one_index = one_index[:,((args.index-1)*one_num):]
	else:
		one_index = one_index[:,((args.index-1)*one_num):(args.index*one_num)]
	# 训练集设置为0
	pre_train_data[one_index[0], one_index[1]] = 0
	
	zero_index = np.where(drug_protein == 0)
	zero_index = np.stack(zero_index).T
	np.random.shuffle(zero_index)
	zero_index = zero_index.T
	zero_num = one_index.shape[1]
	zero_index = zero_index[:,:zero_num]
	
	test_label = np.concatenate((np.ones(one_index.shape[1]),np.zeros(one_index.shape[1])))

	A = np.zeros((5,num,num),dtype=np.int8)
	
	A[0,:drug_num,(drug_num):(drug_num + protein_num)] = pre_train_data
	A[1] = A[0].T
	A[2,:drug_num,:drug_num] = drug_drug
	A[3,(drug_num):(drug_num + protein_num),(drug_num):(drug_num + protein_num)] = protein_protein
	A[4,:drug_num,:drug_num] = drug_sim

	train_label = torch.from_numpy(pre_train_data).float()
	return(A, pro_str, dru_str, train_label, test_label, one_index, zero_index)
