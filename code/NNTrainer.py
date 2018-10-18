import pdb
import numpy as np
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Trainer:
	"""
	"""

	def __init__(self, model, optimizer, 
				 loader_train, loader_val, loader_dev_enroll, loader_dev_test, 
				 train_size, val_size, dev_enroll_size, dev_test_size, 
				 dev_trials, dev_labels,
				 epochs, num_frames, frame_length, 
				 save_models=False):

		self.model = model
		self.optimizer = optimizer

		self.loader_train = loader_train
		self.loader_val = loader_val
		self.loader_dev_enroll = loader_dev_enroll
		self.loader_dev_test = loader_dev_test
		
		self.train_size = train_size
		self.val_size = val_size
		self.dev_enroll_size = dev_enroll_size
		self.dev_test_size = dev_test_size

		self.dev_trials = dev_trials
		self.dev_labels = dev_labels
		
		self.epochs = epochs
		self.num_frames = num_frames
		self.frame_length = frame_length

		self.save_models = save_models


	def runDevEvaluation(self):
		enroll_embeddings_batch = torch.FloatTensor([]).cuda()
		test_embeddings_batch = torch.FloatTensor([]).cuda()

		for batch_num, enroll_utter_batch in enumerate(self.loader_dev_enroll):
			enroll_utter_batch = Variable(enroll_utter_batch.view(-1, 1, self.num_frames, self.frame_length)).cuda()
			output_batch = self.model(enroll_utter_batch, "eval")
			# pdb.set_trace()
			enroll_embeddings_batch = torch.cat((enroll_embeddings_batch, output_batch.data), 0)


		for batch_num, test_utter_batch in enumerate(self.loader_dev_test):
			test_utter_batch = Variable(test_utter_batch.view(-1, 1, self.num_frames, self.frame_length)).cuda()
			output_batch = self.model(test_utter_batch, "eval")
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

		scores_list = [F.cosine_similarity(enroll_embeddings_batch[enroll_index], test_embeddings_batch[test_index], dim=0)[0] for (enroll_index, test_index) in self.dev_trials]

		scores_list = np.array(scores_list)
		err_val, threshold = utils.EER(self.dev_labels, scores_list)
		print("Dev Evaluation - ERR: {} \tThreshold: {}".format(err_val, threshold))
		return err_val


	def runValidation(self):
		iteration_num = 0
		validation_loss = 0
		validation_correct = 0

		for batch_num, (data_batch, target_batch) in enumerate(self.loader_val):
			iteration_num += 1
			data_batch = Variable(data_batch.view(-1, 1, self.num_frames, self.frame_length)).cuda()
			target_batch = Variable(target_batch.view(len(target_batch))).cuda()
			output_batch = self.model(data_batch, "train")

			criterion = nn.CrossEntropyLoss()
			loss = criterion(output_batch, target_batch)
			validation_loss += loss.data[0] # .item()

			pred = output_batch.max(1, keepdim=True)[1]
			validation_correct += pred.eq(target_batch.view_as(pred)).sum() # add .item()
			percentage_complete = (float(iteration_num) * float(data_batch.shape[0])) / float(self.val_size) * 100.0

			print("Validation - Percentage Complete: {}/{} ({:.1f}%)".format(iteration_num * data_batch.shape[0], self.val_size, percentage_complete))

		validation_loss /= float(len(self.loader_val.dataset))
		validation_accuracy = 100.0 * float(validation_correct.data[0]) / float(len(self.loader_val.dataset))
		print("Validation - Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(validation_loss, validation_correct.data[0], len(self.loader_val.dataset), validation_accuracy))
		return validation_accuracy


	def train(self):
		"""	"""
		self.model.train()

		for epoch_num in range(self.epochs):
			iteration_num = 0
			train_loss = 0
			train_correct = 0

			for batch_num, (data_batch, target_batch) in enumerate(self.loader_train):
				iteration_num += 1
				self.optimizer.zero_grad()

				data_batch = Variable(data_batch.view(-1, 1, self.num_frames, self.frame_length)).cuda()
				target_batch = Variable(target_batch.view(len(target_batch))).cuda()
				output_batch = self.model(data_batch, "train")

				criterion = nn.CrossEntropyLoss()
				loss = criterion(output_batch, target_batch)

				loss.backward()
				self.optimizer.step()

				pred = output_batch.data.max(1, keepdim=True)[1]
				predicted = pred.eq(target_batch.data.view_as(pred)).sum() # add .item()
				train_correct += predicted
				train_loss += loss.data[0] #.item()
				percentage_complete = (float(iteration_num) * float(data_batch.shape[0])) / float(self.train_size) * 100.0

				# if iteration_num % 100 == 0:
				print("Training - Epoch: {} \t Percentage Complete: {}/{} ({:.1f}%) \t Loss: {:.8f}".format(epoch_num, iteration_num * data_batch.shape[0], self.train_size, percentage_complete, loss.data[0]))

			validation_accuracy = self.runValidation()
			err = self.runDevEvaluation()

			# Saving model after every epoch
			if self.save_models == True:
				torch.save(self.model.state_dict(), "models/model_train_1-2-3_epoch_{}_ERR_{:.5f}.pkl".format(80+epoch_num, err))
				print("Model saved at models/model_epoch_{}.pkl".format(epoch_num))

