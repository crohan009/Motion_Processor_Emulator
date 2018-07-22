import torch
import torch.nn as nn
import os
# import numpy as np
# import math
# from torch.autograd import Variable
# from torch.utils.data import Dataset
# import torch.optim as optim 


class Network3(nn.Module):	# W.I.P.

	def __init__(self, init_weights=False):
		super(Network3, self).__init__()

		self.conv1 = nn.Conv2d(1, 6, kernel_size=(5,2), stride=(2,1), padding=(1,1), dilation=1, bias=True)
		self.batchnorm1 = nn.BatchNorm2d(6)
		self.conv2 = nn.Conv2d(6, 20, kernel_size=(5,2), stride=(2,1), padding=0, dilation=1, bias=True)
		self.batchnorm2 = nn.BatchNorm2d(20)
		self.conv3 = nn.Conv2d(20, 30, kernel_size=(3,3), stride=(2,2), padding=0, dilation=1, bias=True)
		self.batchnorm3 = nn.BatchNorm2d(30)
		self.conv4 = nn.Conv2d(30, 40, kernel_size=(2,2), stride=(2,2), padding=0, dilation=1, bias=True)
		self.batchnorm4 = nn.BatchNorm2d(40)
		self.conv5 = nn.Conv2d(40, 50, kernel_size=(2,2), stride=1, padding=0, dilation=1, bias=True)
		# self.batchnorm5 = nn.BatchNorm2d(50)

		self.fc1 = nn.Linear(50, 20)		
		self.fc2 = nn.Linear(20, 9)
		self.fc3 = nn.Linear(9, 3)
        
		self.dp1 = nn.Dropout()
		#self.dp3 = nn.Dropout2d()       
		#self.dp3 = nn.Dropout3d()
        
		self.relu = nn.ReLU(inplace=True)

		if(init_weights):
			self.init_weights()

	def forward(self, X):
        
		fwd_map = self.dp1(X)
        
		fwd_map = self.conv1(fwd_map)
		fwd_map = self.batchnorm1(fwd_map)
		self.relu(fwd_map)

		fwd_map = self.conv2(fwd_map)
		fwd_map = self.batchnorm2(fwd_map)
		self.relu(fwd_map)

		fwd_map = self.conv3(fwd_map)
		fwd_map = self.batchnorm3(fwd_map)
		self.relu(fwd_map)

		fwd_map = self.conv4(fwd_map)
		fwd_map = self.batchnorm4(fwd_map)
		self.relu(fwd_map)

		fwd_map = self.conv5(fwd_map)
		# fwd_map = self.batchnorm5(fwd_map)
		self.relu(fwd_map)
        
		fwd_map = self.dp1(fwd_map)

		fwd_map = torch.squeeze(fwd_map)

		fwd_map = self.relu(self.fc1(fwd_map))
		fwd_map = self.relu(self.fc2(fwd_map))
		fwd_map = self.relu(self.fc3(fwd_map))

		return fwd_map.view(1,3)

	def save_checkpoint(self, relative_path, val):
		c_wd = os.getcwd()
		abs_path = c_wd + relative_path

		if not os.path.exists(abs_path):    # if path doesn't exist, create it
			os.makedirs(abs_path)   

		with open(abs_path + '/network_state_checkpoint{}.pth'.format(val), 'wb') as f: 
			torch.save(self.state_dict(), f)

	def load_from_checkpoint(self, relative_path):
		'''
		relative_path: type: string ~> provide a path to the directory to load the pre-trained parameters
		will append the current working directory to relative_path

		torch.load(...,os.getcwd() + relative_path)

		'''
		c_wd = os.getcwd()
		abs_path = c_wd + relative_path

		if not os.path.exists(abs_path):
			print("Path containing pre-trained model weights does not exist! \n")
			return 0

		with open(abs_path, 'rb') as f:
			if torch.cuda.is_available():
				#use the first gpu
				print ("Loading model from \n{} \n...onto the first gpu.\n".format(abs_path))
				return torch.load(abs_path,map_location=lambda storage,loc:storage.cuda(1))
			else:
				print("Loading model from \n{} \n...onto the CPU. \n".format(abs_path))
				return torch.load(abs_path)                



	def init_weights(self):
		layers = list(self.children())
		for i in range(len(layers)):
			if isinstance(layers[i], nn.Conv2d) or isinstance(layers[i], nn.Linear):
				nn.init.kaiming_normal(layers[i].weight)
				if layers[i].bias is not None:
					layers[i].bias.data.zero_()
			elif isinstance(layers[i], nn.BatchNorm2d):
				layers[i].weight.data.fill_(1)
				layers[i].bias.data.zero_()
