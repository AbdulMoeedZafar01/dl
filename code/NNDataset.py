import numpy as np
import pdb

import torch
from torch.utils.data.dataset import Dataset

class DatasetTrain(Dataset):
	""" 
	"""

	def __init__(self, features, labels, padding_size):
		""" """
		self.features = features
		self.labels = labels
		self.padding_size = padding_size


	def __getitem__(self, index):
		""" """

		# Extract utterance from the features set
		utterance = self.features[index]

		# Pad utterances up and down with wrapping and size equal padding_size 
		padded_utterance = np.pad(utterance, ((self.padding_size, self.padding_size), (0, 0)), 'wrap')

		# Generate random frame index inside an utterance
		frame_index = np.random.randint(low=0+self.padding_size, high=utterance.shape[0]+self.padding_size)

		# Extract out the context
		context = np.array(padded_utterance[frame_index - self.padding_size : frame_index + self.padding_size + 1].tolist())

		# Create tensor item and label to return
		item = torch.from_numpy(context).float()
		label = torch.LongTensor([int(self.labels[index])])
		return item, label


	def __len__(self):
		""" """
		return len(self.labels)


class DatasetValidation(Dataset):
	""" 
	This is the Dataset class implementation for validation dataset that is cropped out of the training dataset for learning embeddings
	"""

	def __init__(self, features, labels, padding_size):
		""" """
		self.features = features
		self.labels = labels
		self.padding_size = padding_size


	def __getitem__(self, index):
		""" """

		# Extract utterance from the features set
		utterance = self.features[index]

		# Pad utterances up and down with wrapping and size equal padding_size 
		padded_utterance = np.pad(utterance, ((self.padding_size, self.padding_size), (0, 0)), 'wrap')

		# Generate random frame index inside an utterance
		frame_index = np.random.randint(low=0+self.padding_size, high=utterance.shape[0]+self.padding_size)

		# Extract out the context
		context = np.array(padded_utterance[frame_index - self.padding_size : frame_index + self.padding_size + 1].tolist())

		# Create tensor item and label to return
		item = torch.from_numpy(context).float()
		label = torch.LongTensor([int(self.labels[index])])
		return item, label


	def __len__(self):
		""" """
		return len(self.labels)


class DatasetDevEnroll(Dataset):
	""" 
	"""

	def __init__(self, enrollment_utters, padding_size):
		""" """
		self.enrollment_utters = enrollment_utters
		self.padding_size = padding_size


	def __getitem__(self, index):
		""" """

		# Create enrollment utterance item
		enroll_utterance = self.enrollment_utters[index]
		padded_enroll_utterance = np.pad(enroll_utterance, ((self.padding_size, self.padding_size), (0, 0)), 'wrap')
		frame_index = np.random.randint(low=0+self.padding_size, high=enroll_utterance.shape[0]+self.padding_size)
		context_enroll_utter = np.array(padded_enroll_utterance[frame_index - self.padding_size : frame_index + self.padding_size + 1].tolist())
		item_enroll_utter = torch.from_numpy(context_enroll_utter).float()

		return item_enroll_utter


	def __len__(self):
		""" """
		return len(self.enrollment_utters)


class DatasetDevTest(Dataset):
	""" 
	"""

	def __init__(self, test_utters, padding_size):
		""" """
		self.test_utters = test_utters
		self.padding_size = padding_size


	def __getitem__(self, index):
		""" """

		# Create test utterance item
		test_utterance = self.test_utters[index]
		padded_test_utterance = np.pad(test_utterance, ((self.padding_size, self.padding_size), (0, 0)), 'wrap')
		frame_index = np.random.randint(low=0+self.padding_size, high=test_utterance.shape[0]+self.padding_size)
		context_test_utter = np.array(padded_test_utterance[frame_index - self.padding_size : frame_index + self.padding_size + 1].tolist())
		item_test_utter = torch.from_numpy(context_test_utter).float()

		return item_test_utter


	def __len__(self):
		""" """
		return len(self.test_utters)


class DatasetTestEnroll(Dataset):
	""" 
	"""

	def __init__(self, enrollment_utters, padding_size):
		""" """
		self.enrollment_utters = enrollment_utters
		self.padding_size = padding_size


	def __getitem__(self, index):
		""" """

		# Create enrollment utterance item
		enroll_utterance = self.enrollment_utters[index]
		padded_enroll_utterance = np.pad(enroll_utterance, ((self.padding_size, self.padding_size), (0, 0)), 'wrap')
		frame_index = np.random.randint(low=0+self.padding_size, high=enroll_utterance.shape[0]+self.padding_size)
		context_enroll_utter = np.array(padded_enroll_utterance[frame_index - self.padding_size : frame_index + self.padding_size + 1].tolist())
		item_enroll_utter = torch.from_numpy(context_enroll_utter).float()

		return item_enroll_utter


	def __len__(self):
		""" """
		return len(self.enrollment_utters)


class DatasetTestTest(Dataset):
	""" 
	"""

	def __init__(self, test_utters, padding_size):
		""" """
		self.test_utters = test_utters
		self.padding_size = padding_size


	def __getitem__(self, index):
		""" """

		# Create test utterance item
		test_utterance = self.test_utters[index]
		padded_test_utterance = np.pad(test_utterance, ((self.padding_size, self.padding_size), (0, 0)), 'wrap')
		frame_index = np.random.randint(low=0+self.padding_size, high=test_utterance.shape[0]+self.padding_size)
		context_test_utter = np.array(padded_test_utterance[frame_index - self.padding_size : frame_index + self.padding_size + 1].tolist())
		item_test_utter = torch.from_numpy(context_test_utter).float()

		return item_test_utter


	def __len__(self):
		""" """
		return len(self.test_utters)

