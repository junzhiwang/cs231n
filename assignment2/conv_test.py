import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np

import timeit

dtype = torch.FloatTensor # the CPU datatype

class Flatten(nn.Module):
	def forward(self, x):
		N, C, H, W = x.size()  # read in N, C, H, W
		return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


in_channel, out_channel, kernel_size, stride = 3, 32, 7, 1
pool_size, pool_stride = 2, 2
affine_in, affine_out = 5408, 1024
affine_in_2, affine_out_2 = 1024, 10

fixed_model_base = nn.Sequential(
	nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride),
	nn.ReLU(),
	nn.BatchNorm2d(out_channel, affine=True),
	nn.MaxPool2d(kernel_size=pool_size),
	Flatten(),
	nn.Linear(affine_in, affine_out),
	nn.ReLU(),
	nn.Linear(affine_in_2, affine_out_2)
)

fixed_model_base.type(dtype)
loss_fn = nn.CrossEntropyLoss().type(dtype)
optimizer = optim.RMSprop(fixed_model_base.parameters())