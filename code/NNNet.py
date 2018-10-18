import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential



class Flatten(nn.Module):
	"""
	A simple custom module that reshapes (n, m, 1, 1) tensors to (n, m).
	"""

	def forward(self, input):
		return input.view(input.size(0), -1)


class ResnetBasicBlock(nn.Module):
	"""
	"""

	def __init__(self, in_planes, out_planes):
		super(ResnetBasicBlock, self).__init__()

		self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(num_features=out_planes)
		self.elu1 = nn.ELU(inplace=True)
		
		self.conv2 = nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(num_features=out_planes)
		self.elu2 = nn.ELU()
		

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.elu1(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out += residual
		out = self.elu2(out)

		return out


class Net(nn.Module):
	"""
	"""

	def __init__(self, in_channels, num_speakers, num_embeddings=256):
		super(Net, self).__init__()

		self.layer1_conv5x5 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=2, padding=0, bias=False)
		self.layer2_elu = nn.ELU()
		self.layer3_resblock = ResnetBasicBlock(in_planes=32, out_planes=32)

		self.layer4_conv5x5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=0, bias=False)
		self.layer5_elu = nn.ELU()
		self.layer6_resblock = ResnetBasicBlock(in_planes=64, out_planes=64)

		self.layer7_conv5x5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=0, bias=False)
		self.layer8_elu = nn.ELU()
		self.layer9_resblock = ResnetBasicBlock(in_planes=128, out_planes=128)

		self.layer10_conv5x5 = nn.Conv2d(in_channels=128, out_channels=num_embeddings, kernel_size=5, stride=2, padding=0, bias=False)
		self.layer11_elu = nn.ELU()
		self.layer12_resblock = ResnetBasicBlock(in_planes=num_embeddings, out_planes=num_embeddings)

		self.layer_flatten = Flatten()

		self.layer_linear = nn.Linear(num_embeddings, num_speakers)


	def forward(self, x, train_mode):

		out = self.layer1_conv5x5(x)
		out = self.layer2_elu(out)
		out = self.layer3_resblock(out)

		out = self.layer4_conv5x5(out)
		out = self.layer5_elu(out)
		out = self.layer6_resblock(out)

		out = self.layer7_conv5x5(out)
		out = self.layer8_elu(out)
		out = self.layer9_resblock(out)

		out = self.layer10_conv5x5(out)
		out = self.layer11_elu(out)
		out = self.layer12_resblock(out)

		out = F.avg_pool2d(out, kernel_size=(out.shape[2], 1))
		out = F.avg_pool2d(out, kernel_size=(1, out.shape[3]))

		out = self.layer_flatten(out)

		out = F.normalize(out, p=2, dim=1)

		if train_mode == "train":
			out = self.layer_linear(out)
		
		return out



def initWeights(m):
	"""
	"""
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight.data)

	if isinstance(m, nn.Linear):
		init.xavier_uniform(m.weight.data)
