import torch.nn as nn

class Network1(nn.Module):

	def __init__(self, init_weights=False):
		super(Network1, self).__init__()

		# Inputs = 9; Outputs = 3; Hidden = (3 layers) 20, 30 ,10
		self.fc1 = nn.Linear(9, 30)		# 		input layer
		self.fc2 = nn.Linear(30, 30)	#	-
		self.fc3 = nn.Linear(30, 30)	#	 |-	hidden layers
		self.fc4 = nn.Linear(30, 3)		#	-
		self.relu = nn.ReLU(inplace=True)

		if(init_weights):
			self.init_weights()

	def forward(self, x):
		x = self.relu(self.fc1(x))
		x = self.relu(self.fc2(x))
		x = self.relu(self.fc3(x))
		x = self.relu(self.fc4(x))
		return x

	def init_weights(self):
		layers = list(self.children())
		for i in range(len(layers)):
			if isinstance(layers[i], nn.Linear):
				nn.init.normal(layers[i].weight)
				nn.init.normal(layers[i].bias)
