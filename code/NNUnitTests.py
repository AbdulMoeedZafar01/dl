import numpy as np
import pdb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import NNDataset
import NNNet

def testTrainData():
	features_test = np.array([np.array([[1, 2, 3], [4, 5, 6]]), np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]]), np.array([[-1, -2, -3]])])
	labels_test = np.array([0, 2, 1])
	
	dataset_train = NNDataset.DatasetTrain(features_test, labels_test, 4)
	loader_train = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=1)
	
	for batch_idx, (data_batch, target_batch) in enumerate(loader_train):
		print(target_batch)
		print(data_batch)
		pdb.set_trace()


def testNeuralNetModel():

	model = NNNet.Net(in_channels=1, num_speakers=10, num_embeddings=256)
	# model.apply(NNNet.initWeightsXavier)

	input_feature = torch.randn(2, 5000, 64)
	input_feature = Variable(input_feature.view(-1, 1, 5000, 64))
	output = model(input_feature, "train")

	pdb.set_trace()
