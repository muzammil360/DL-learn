# This file is responsible for training the network
import sys
sys.path.append('..')

import torch


from src.data import getDataLoader
from src.model import getModel
from src.optimizer import getOptimizer
from src.loss import getLossFunc

def train(_data_loader, _model, _optimizer, _loss):
	pass


def main():

	model_identifier = 'alexnet'

	data_loader = getDataLoader()
	model = getModel(model_identifier)
	optimizer = getOptimizer()
	loss = getLossFunc()


	x = torch.rand(1, 3,224,224)
	y = model(x)
	print('output shape: {}'.format(y.shape))


	train(data_loader, model, optimizer, loss)

if __name__=="__main__":
	main()