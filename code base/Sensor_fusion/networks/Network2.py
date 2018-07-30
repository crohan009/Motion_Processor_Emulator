import torch
import torch.nn as nn
import os
import numpy as np
import math
from torch.autograd import Variable
from torch.utils.data import Dataset
import torch.optim as optim 
import random
from sklearn.metrics import r2_score, explained_variance_score
from matplotlib import pyplot as plt


class Network2(nn.Module):	# W.I.P.

    def __init__(self, init_weights=False):
        super(Network2, self).__init__()

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

        self.relu = nn.ReLU(inplace=True)

        if(init_weights):
            self.init_weights()

    def forward(self, X):
        fwd_map = self.conv1(X)
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
                
                
    def training(self, epochs, dsets_enqueuer_training, dsets_enqueuer_testing):
        # Loss Function(s)
        # criterion = nn.MSELoss()             # Mean Squared Error Loss (L2 Loss)
        # criterion = nn.L1Loss()              # Mean Absolute Error Loss (L1 Loss)
        # criterion = nn.SmoothL1Loss()        # Huber Loss 
        criterion = lambda x, target: torch.mul((1 - torch.cos(target - x)), (1 - torch.cos(target - x)))

        # Optimizer
        optimizer = optim.Adam(self.parameters(),lr = 0.0001, betas=(0.9, 0.999), eps=1e-08)
        # optimizer = optim.Adagrad(model.parameters(), lr=0.0001, lr_decay=0, weight_decay=0)

        if torch.cuda.is_available():
            criterion = criterion.cuda()

        # Variables to keep track of losses 
        loss_data_training = 0.0
        loss_data_testing = 0.0

        loss_lst_train = []
        loss_lst_test = []

        r_sq_lst = []
        r_sq_lst_test = []

        var_exp_lst = []
        var_exp_lst_test = []
        
        print("\n\n\n[ Training Network ]\n")

        for Epoch in range(epochs):

            y_per_epoch = []
            output_per_epoch = []

            #############################################################################################################
            #                                                  TRAINING                                                 #
            #############################################################################################################

            for idx, data in enumerate(dsets_enqueuer_training, 1):
                x,y = data['x'], data['y']

                if torch.cuda.is_available():
                    x, y = Variable(x.cuda(), requires_grad = True).float(), Variable(y.cuda(), requires_grad = True).float()
                else:
                    x, y = Variable(x, requires_grad = True).float(), Variable(y, requires_grad = True).float()

                # print(x.shape)
                # print(y.shape)

                # break

                self.train()
                output = self(x)

                # print(output.shape, type(output), output.data.numpy(), type(output.data.numpy()))
                # print(y.shape, type(y), y.data.numpy(), type(y.data.numpy()))

                optimizer.zero_grad()
                loss = criterion(output * np.pi/180 , y * np.pi/180)
                loss.backward(torch.Tensor([1, 1, 1]))
                optimizer.step()

                loss_data_training += loss.data

                y_per_epoch.append(y.data.numpy())
                output_per_epoch.append(output.data.numpy())

            R_sq_score = r2_score(np.squeeze(np.array(y_per_epoch)), 
                                  np.squeeze(np.array(output_per_epoch))) 

            var_exp = explained_variance_score(np.squeeze(np.array(y_per_epoch)), 
                                               np.squeeze(np.array(output_per_epoch)))  

            if((Epoch+1) % 1 == 0):
                print ("Epoch \t{0} / {2} , \t loss = {1} , \t R^2 = {3} , \t Var exp = {4} %".format( Epoch+1,
                                                                                        loss_data_training.cpu().numpy()/idx , 
                                                                                             epochs, 
                                                                                             round(R_sq_score, 4),
                                                                                             round(var_exp*100, 4) ))


            loss_lst_train.append(loss_data_training.cpu().numpy()/idx)
            r_sq_lst.append(R_sq_score)
            var_exp_lst.append(var_exp)

            loss_data_training = 0.0

            y_per_epoch = []
            output_per_epoch = []

            #############################################################################################################
            #                                                  TESTING                                                  #
            #############################################################################################################


            for idx, data in enumerate(dsets_enqueuer_testing, 1):
                x,y = data['x'], data['y']

                if torch.cuda.is_available():
                    x, y = Variable(x.cuda(), requires_grad = True).float(), Variable(y.cuda(), requires_grad = True).float()
                else:
                    x, y = Variable(x, requires_grad = True).float(), Variable(y, requires_grad = True).float()

                self.eval()
                output = self(x)
                loss = criterion(output * np.pi/180 , y * np.pi/180 )
                loss_data_testing += loss.data

                y_per_epoch.append(y.data.numpy())
                output_per_epoch.append(output.data.numpy())

            R_sq_score = r2_score(np.squeeze(np.array(y_per_epoch)), 
                                  np.squeeze(np.array(output_per_epoch))) 

            var_exp = explained_variance_score(np.squeeze(np.array(y_per_epoch)), 
                                               np.squeeze(np.array(output_per_epoch))) 

            loss_lst_test.append(loss_data_testing.cpu().numpy()/idx)
            r_sq_lst_test.append(R_sq_score)
            var_exp_lst_test.append(var_exp)
            loss_data_testing = 0.0

            if(Epoch > 0 and Epoch%10 == 0):                                     # Saving network weights every 10 Epochs
                model.save_checkpoint("/saved_model_weights/", Epoch)

            y_per_epoch = []
            output_per_epoch = []


        print("\n\n[ Training Complete ]\n")      

        print("\n\nMax training R squared value = ", max(r_sq_lst))
        print("Max training variance explained = ", max(var_exp_lst))

        print("\n\nMax testing R squared value = ", max(r_sq_lst_test))
        print("Max testing variance explained = ", max(var_exp_lst_test))
        
        return loss_lst_train, loss_lst_test, r_sq_lst, r_sq_lst_test, var_exp_lst, var_exp_lst_test
        
                
                
                
class Network2_Data_loader(Dataset):
    # Loads data into the Network2 model

    def __init__(self, abs_filename, time_window=50, trans=False, sequential_test_mode=False):
        data, self.runtime = self.get_data(abs_filename)
        if self.runtime == 0:
            self.runtime = None
        try:
            self.data_matrix = data.reshape((1,data.shape[0], data.shape[1]))
            #print(self.data_matrix.shape)
        except IndexError:
            print("Data Error!: Data shape: ",data.shape)

        self.tw = time_window          # time window of samples
        
        self.rand_sel_lst = list(range(0,len(self))) 
        
        if(not sequential_test_mode):             # if you don't want the data to be loaded in the original order
            random.shuffle(self.rand_sel_lst)
        
        if(trans):
            self.transform_data()

    def get_data(self, abs_filename):
        # File specific parser for extracting sensor data
        file = open(abs_filename, 'r')
        run_time = 0                                    # program runtime (in seconds)
        data = []

        for line in file:

            #if line.count('\t') != 14:                  # skip line if data is not nominal
            #    continue

            if line[:10] == 'Start_time':
                run_time = self.get_runtime(line)
                continue

            line_lst = line.rstrip().split("\t")
            data_lst = line_lst[1:10] + line_lst[-3:]

            #if not self.isValidEntry(data_lst):         # skip line if data is not nominal
            #    continue

            if len(data_lst) == 12:
                data_lst[8] = data_lst[8][:-1]
                try :
                    for i in range(len(data_lst)):
                        data_lst[i] = float(data_lst[i])
                except ValueError:
                    continue
                data.append(data_lst)

        return np.array(data), run_time 
    
    def get_runtime(self, line):
        line_lst = line.rstrip().split(';')
        start_time, end_time = line_lst[0][-12:], line_lst[1][-12:]
        start_time, end_time = start_time.split(':'), end_time.split(':')
        lst = []
        for i in zip(start_time, end_time):
            a, b = float(i[0]), float(i[1])
            lst.append(b-a)
        return( 3600*lst[0] + 60*lst[1] + lst[2] )

    def __len__(self):
        # return math.ceil(self.data_matrix.shape[1] / self.tw)
        #return (self.data_matrix.shape[1] // self.tw)
        return (self.data_matrix.shape[1] - self.tw)

    def transform_data(self):
        trans = np.array([-1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, 1]).reshape((1,1,12)) # data specific transformation
        return self.data_matrix * trans


    def __getitem__(self,idx, ):        # idx ranges from 0 to len(self) 

        # if (idx+1)*self.tw > self.data_matrix.shape[1]:
        # 	d = self.data_matrix[:, idx*self.tw: , :]
        # else:
        i = self.rand_sel_lst[idx]
        #d = self.data_matrix[:, i*self.tw:(i+1)*self.tw , :]
        d = self.data_matrix[:, i:i+self.tw , :]

        return {'x': d[:, :, :9], 'y': d[:, -1:, 9:].squeeze() }                


