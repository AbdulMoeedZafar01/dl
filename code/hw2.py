import os
import logging
import numpy as np
import pdb
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable

import utils
import NNDataset
import NNNet
import NNUnitTests
import NNTrainer

class Files:
	"""
	"""

	def __init__(self):
		self.train_dir = "data/"
		self.train_files = [1, 2, 3]
		self.dev_file = "data/dev.preprocessed.npz"
		self.test_file = "data/test.preprocessed.npz"

		self.final_model_dir = "models/final_model.pkl"
		self.score_dir = "scores/scores.npy"


class Parameters:
	"""
	"""

	def __init__(self):
		self.padding_size = 3000
		self.batch_size = 40
		self.epochs = 40
		self.learning_rate = 0.001 # Previous (most recent to the left): 0.01
		self.momentum = 0.9
		self.weight_decay = 0.000001
		self.num_workers = 1

		self.num_embeddings = 256


def partitionTrainingValidationData(features, labels):
	""" """
	total_data_length = features.shape[0]
	training_data_length = int(0.95 * total_data_length)

	train_features = features[0:training_data_length]
	train_labels = labels[0:training_data_length]

	val_features = features[training_data_length:]
	val_labels = labels[training_data_length:]

	return (train_features, train_labels, val_features, val_labels)


def runTestPrediction(model, loader_test_enroll, loader_test_test, test_trials, num_frames, frame_length):
	print("Making prediction on test data...")
	enroll_embeddings_batch = torch.FloatTensor([]).cuda()
	test_embeddings_batch = torch.FloatTensor([]).cuda()

	for batch_num, enroll_utter_batch in enumerate(loader_test_enroll):
		enroll_utter_batch = Variable(enroll_utter_batch.view(-1, 1, num_frames, frame_length)).cuda()
		output_batch = model(enroll_utter_batch, "eval")
		# pdb.set_trace()
		enroll_embeddings_batch = torch.cat((enroll_embeddings_batch, output_batch.data), 0)


	for batch_num, test_utter_batch in enumerate(loader_test_test):
		test_utter_batch = Variable(test_utter_batch.view(-1, 1, num_frames, frame_length)).cuda()
		output_batch = model(test_utter_batch, "eval")
		test_embeddings_batch = torch.cat((test_embeddings_batch, output_batch.data), 0)

	dev_correct = 0
	iteration_num = 0
	scores_list = []
	
	# Iterating over trials and determining similarity scores
	# for (enroll_index, test_index) in self.dev_trials:
	# 	enroll_embedd = enroll_embeddings_batch[enroll_index]
	# 	test_embedd = test_embeddings_batch[test_index]

	# 	score = F.cosine_similarity(enroll_embedd, test_embedd, dim=0) # .item()
	# 	scores_list.append(score[0])

	scores_list = [F.cosine_similarity(enroll_embeddings_batch[enroll_index], test_embeddings_batch[test_index], dim=0)[0] for (enroll_index, test_index) in test_trials]
	scores = np.array(scores_list)
	return scores


def main():

	# Starting main...
	print("AM: Starting program")

	files = Files()

	# Load data
	# all_train_features, all_train_labels, num_speakers = utils.train_load(files.train_dir, files.train_files)
	num_speakers = 889
	# train_features, train_labels, val_features, val_labels = partitionTrainingValidationData(all_train_features, all_train_labels)
	# dev_trials, dev_labels, dev_enroll_utters, dev_test_utters = utils.dev_load(files.dev_file)
	test_trials, test_enroll_utters, test_test_utters = utils.test_load(files.test_file)
	print("AM: Data loaded.")

	# Initializing constants
	params = Parameters()
	num_frames = (2 * params.padding_size + 1)
	frame_length = 64
	output_size = num_speakers
	print("AM: Parameters initialized.")

	# Create 'Dataset's
	# dataset_train = NNDataset.DatasetTrain(train_features, train_labels, params.padding_size)
	# dataset_val = NNDataset.DatasetValidation(val_features, val_labels, params.padding_size)
	# dataset_dev_enroll = NNDataset.DatasetDevEnroll(dev_enroll_utters, params.padding_size)
	# dataset_dev_test = NNDataset.DatasetDevTest(dev_test_utters, params.padding_size)
	dataset_test_enroll = NNDataset.DatasetTestEnroll(test_enroll_utters, params.padding_size)
	dataset_test_test = NNDataset.DatasetTestTest(test_test_utters, params.padding_size)
	# dataset_test = 
	print("AM: Dataset's created.")

	# Create 'DataLoader's
	# loader_train = DataLoader(dataset_train, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
	# loader_val = DataLoader(dataset_val, batch_size=params.batch_size, shuffle=True, num_workers=params.num_workers)
	# loader_dev_enroll = DataLoader(dataset_dev_enroll, batch_size=20, shuffle=False, num_workers=params.num_workers)
	# loader_dev_test = DataLoader(dataset_dev_test, batch_size=20, shuffle=False, num_workers=params.num_workers)

	loader_test_enroll = DataLoader(dataset_test_enroll, batch_size=1, shuffle=False, num_workers=params.num_workers)
	loader_test_test = DataLoader(dataset_test_test, batch_size=1, shuffle=False, num_workers=params.num_workers)

	# loader_test = 
	print("AM: DataLoader's created.")

	# Create neural net model
	model = NNNet.Net(in_channels=1, num_speakers=num_speakers, num_embeddings=params.num_embeddings).cuda()
	model.load_state_dict(torch.load(files.final_model_dir))
	print("AM: Neural net model created.")

	# Create optimizer
	optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate, momentum=params.momentum, weight_decay=params.weight_decay)
	print("AM: Optimizer created.")

	# Create trainer
	# trainer = NNTrainer.Trainer(model, 
	# 							optimizer, 
	# 							loader_train, 
	# 							loader_val, 
	# 							loader_dev_enroll,
	# 							loader_dev_test,
	# 							len(dataset_train), 
	# 							len(dataset_val), 
	# 							len(dataset_dev_enroll),
	# 							len(dataset_dev_test),
	# 							dev_trials,
	# 							dev_labels,
	# 							params.epochs,
	# 							num_frames,
	# 							frame_length,
	# 							True)
	# trainer.train()
	print("AM: Training done.")

	# Save model
	# torch.save(model.state_dict(), files.final_model_dir)
	print("AM: Final model saved.")

	# Make predictions
	# makePredictionsOnTest(model, loader_test, len(dataset_test), files.score_dir)
	scores = runTestPrediction(model, loader_test_enroll, loader_test_test, test_trials, num_frames, frame_length)
	np.save(files.score_dir, scores)
	print("AM: Predictions done.")


if __name__ == '__main__':
	# NNUnitTests.testTrainData()
	# NNUnitTests.testNeuralNetModel()
	main()

