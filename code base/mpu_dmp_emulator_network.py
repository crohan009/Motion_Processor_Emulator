import torch
import torch.nn as nn
# import os
import numpy as np
import math
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.optim as optim 

from sklearn.metrics import r2_score, explained_variance_score

# from ./networks/Network1 import Network1
from networks.Network2 import Network2

class Data_loader(Dataset):
	# Loads data into the model

	def __init__(self, abs_filename, time_window=50, trans=False):
		data = self.get_data(abs_filename)

		self.data_matrix = data.reshape((1,data.shape[0], data.shape[1]))
		self.tw = time_window							# time window of samples

		if (trans):
			self.data_matrix = self.transform_data()

	def get_data(self, abs_filename):				
		# File specific parser for extracting sensor data
		file = open(abs_filename, 'r')
		data = []
		for line in file:
			line_lst = line.rstrip().split("\t")
			data_lst = line_lst[1:10] + line_lst[-3:]
			data_lst[8] = data_lst[8][:-1]
			for i in range(len(data_lst)):
				data_lst[i] = float(data_lst[i])
			if len(data_lst) == 12:
				data.append(data_lst)
		return np.array(data)

	def __len__(self):
		# return math.ceil(self.data_matrix.shape[1] / self.tw)
		return (self.data_matrix.shape[1] // self.tw)

	def transform_data(self):
		trans = np.array([-1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1]).reshape((1,1,12)) # data specific transformation
		return self.data_matrix * trans
		

	def __getitem__(self,idx, ): 		# idx ranges from 0 to len(self) 
		
		# if (idx+1)*self.tw > self.data_matrix.shape[1]:
		# 	d = self.data_matrix[:, idx*self.tw: , :]
		# else:
		d = self.data_matrix[:, idx*self.tw:(idx+1)*self.tw , :]	
			
		return {'x': d[:, :, :9], 'y': d[:, -1:, 9:].squeeze() }	


def data_extractor(filename, processed_filename1, processed_filename2=None):
	# cleans IMU outputs and generates usable dataset as a clean CSV file
	file = open(filename, 'r')
	out_file1 = open(processed_filename1, 'a')
	if processed_filename2 is not None:
		out_file2 = open(processed_filename2, 'a')

	for line in file:
		line_lst = line.rstrip().split("\t")

		if line_lst[0]=='LSM_raw: ' :
			print(line.rstrip(), file=out_file1, end=";\t\t")
		# elif line_lst[0]=='LSM_ypr: ' :
		# 	print(line.rstrip(), file=out_file2, end="\n")

		# elif line_lst[0]=='MPU_raw: ' :
		# 	print(line.rstrip(), file=out_file1, end=";\t\t")
		elif line_lst[0]=='MPU_ypr: ' :
			print(line.rstrip(), file=out_file1, end="\n")
	
	if processed_filename2 is not None:		
		out_file2.close()
	out_file1.close()
	file.close()		


if __name__ == "__main__":

	# Data Pre-processing 
	# data_extractor("i2c/Datasets/not_clean_dataset.txt", "i2c/Datasets/mpu_dataset.txt", "i2c/Datasets/lsm_dataset.txt")
	# data_extractor("MPU_MPL_LM4F_TM4C_uart/Datasets/not_clean_dataset.txt", "MPU_MPL_LM4F_TM4C_uart/Dataset/dataset.txt")
	# data_extractor("MPU_MPL_LM4F_TM4C_uart/Datasets/misc_dataset.txt", "MPU_MPL_LM4F_TM4C_uart/Datasets/misc_dataset_cleaned.txt")

	model = Network2(init_weights=True)
	# print(model)

	dset_train = Data_loader("MPU_MPL_LM4F_TM4C_uart/Datasets/misc_dataset_cleaned.txt", trans=True)
	# print(dset_train.data_matrix[0])
	# print("Shape of dataset = ", dset_train.data_matrix.shape)
	# print("#Samples in dataset = ", len(dset_train))
	# for i in range(len(dset_train)):
	# 	print(dset_train.__getitem__(i)['x'].shape, dset_train.__getitem__(i)['y'].shape)
	dsets_enqueuer_training = torch.utils.data.DataLoader(dset_train, batch_size=1, num_workers=1, drop_last=False)

	# # criterion = nn.BCEWithLogitsLoss()
	criterion = nn.MSELoss()

	optimizer = optim.Adam(model.parameters(),lr = 0.001, betas=(0.9, 0.999), eps=1e-08)

	if torch.cuda.is_available():
		criterion = criterion.cuda()

	loss_data = 0.0
	# # loss_data_testing = 0.0

	# # loss_per_epoch_lst = []

	print("\n\n\n[ Training Network ]\n")

	loss_lst_train = []
	# # loss_lst_test = []

	epochs = 10000

	for Epoch in range(epochs):

		for idx, data in enumerate(dsets_enqueuer_training, 1):
			x,y = data['x'], data['y']

			if torch.cuda.is_available():
				x, y = Variable(x.cuda(), requires_grad = False).float(), Variable(y.cuda(), requires_grad = False).float()
			else:
				x, y = Variable(x, requires_grad = False).float(), Variable(y, requires_grad = False).float()

			# print(x.shape)
			# print(y.shape)

			model.train()
			output = model(x)

			# print(output.shape, type(output), output.data.numpy(), type(output.data.numpy()))
			# print(y.shape, type(y), y.data.numpy(), type(y.data.numpy()))


			optimizer.zero_grad()
			loss = criterion(output,y)
			loss.backward()
			optimizer.step()

			loss_data += loss.data

			R_sq_score = r2_score(y.data.numpy(), output.data.numpy()) 
			var_exp = explained_variance_score(y.data.numpy(), output.data.numpy())  

		print ("Epoch {0} / {2} , \t loss = {1} , \t R^2 = {3} , \t Explained Variance = {4} %".format( Epoch+1, 
																							loss_data.cpu().numpy()/idx , 
																							epochs, 
																							round(R_sq_score, 2),
																							var_exp*100 ))

		loss_lst_train.append(loss_data.cpu().numpy()/idx)	
		loss_data = 0.0	













