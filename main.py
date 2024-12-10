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



if __name__ == '__main__':
	main()

