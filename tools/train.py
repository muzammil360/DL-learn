# This file is responsible for training the network
import sys
sys.path.append('..')

from src.data import getDataLoader
from src.model import getModel
from src.optimizer import getOptimizer
from src.loss import getLossFunc

def train(_data_loader, _model, _optimizer, _loss):
	pass


def main():

	data_loader = getDataLoader()
	model = getModel()
	optimizer = getOptimizer()
	loss = getLossFunc()

	train(data_loader, model, optimizer, loss)

if __name__=="__main__":
	main()