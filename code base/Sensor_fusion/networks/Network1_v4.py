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
# from matplotlib import pyplot as plt


class Network1(nn.Module):

    def __init__(self, init_weights=False):
        super(Network1, self).__init__()

        self.name = "Net1_v4"

        # Inputs = 9; Outputs = 3; Hidden = (3 layers) 20, 30 ,10
        self.fc1 = nn.Linear(9, 7) 
        self.fc2 = nn.Linear(7, 5)   
        self.fc3 = nn.Linear(5, 4)  
        self.fc4 = nn.Linear(4, 3)  
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
                
    
    def train_model(self, epochs, dsets_enqueuer_training, dsets_enqueuer_testing):

        criterion = lambda x, target: torch.mul((1 - torch.cos(target - x)), (1 - torch.cos(target - x)))

        # Optimizer
        optimizer = optim.Adam(self.parameters(),lr = 0.0001, betas=(0.9, 0.999), eps=1e-08)
        # optimizer = optim.Adagrad(self.parameters(), lr=0.0001, lr_decay=0, weight_decay=0)

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
                self.save_checkpoint("/saved_model_weights/{}/".format(self.name), Epoch)

            y_per_epoch = []
            output_per_epoch = []


        print("\n\n[ Training Complete ]\n")      

        print("\n\nMax training R squared value = ", max(r_sq_lst))
        print("Max training variance explained = ", max(var_exp_lst))

        print("\n\nMax testing R squared value = ", max(r_sq_lst_test))
        print("Max testing variance explained = ", max(var_exp_lst_test))
        
        return loss_lst_train, loss_lst_test, r_sq_lst, r_sq_lst_test, var_exp_lst, var_exp_lst_test
        
        
        
        
        
        
        
        
        
        
        
class Network1_Data_loader(Dataset):
    # Loads data into the Network1 model

    def __init__(self, abs_filename, trans=False, sequential_test_mode=True):
        data, self.runtime = self.get_data(abs_filename)
        if self.runtime == 0:
            self.runtime = None
        try:
            self.data_matrix = data.reshape((data.shape[0], data.shape[1]))
        except IndexError:
            print("Data Error!: Data shape: ",data.shape)
        
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
        return (self.data_matrix.shape[0])

    def transform_data(self):
        trans = np.array([-1, -1, 1, 
                          -1, -1, 1, 
                          -1, -1, 1, 1, 1, 1]).reshape((1,1,12)) # data specific transformation
        return self.data_matrix * trans


    def __getitem__(self,idx ):        # idx ranges from 0 to len(self) 

        i = self.rand_sel_lst[idx]
        return {'x': self.data_matrix[i, :9], 'y': self.data_matrix[i, 9:]}

    
    
# if __name__ == "__main__":
    
#     model = Network1(init_weights=True)
#     data_enq_trial = Network1_Data_loader("../Datasets/MPU_rawtoypr_data_cleaned_test_data.csv", sequential_test_mode=True)
#     dsets_enqueuer_trial = torch.utils.data.DataLoader(data_enq_trial, batch_size=1, num_workers=1, drop_last=False)
    
#     y_lst = []
#     o_lst = []
#     data_ctr = 0
    

#     print(data_enq_trial.data_matrix.shape)
#     print(type(data_enq_trial.data_matrix))
#     print(data_enq_trial.rand_sel_lst[:10])
#     print(data_enq_trial.data_matrix.shape[0])
#     print(data_enq_trial.__getitem__(0)['x'])
#     print(data_enq_trial.__getitem__(0)['y'])



#     for idx, data in enumerate(dsets_enqueuer_trial, 1):
#         x,y = data['x'], data['y']

#         if torch.cuda.is_available():
#             x, y = Variable(x.cuda(), requires_grad = False).float(), Variable(y.cuda(), requires_grad = False).float()
#         else:
#             x, y = Variable(x, requires_grad = False).float(), Variable(y, requires_grad = False).float()

#         model.eval()
#         output = model(x)
        
#         y_lst.append(y)
#         o_lst.append(output)
#         data_ctr += 1
#         if data_ctr >= 1000:
#             break
#         break  