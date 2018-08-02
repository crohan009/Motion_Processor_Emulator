import torch
import torch.nn as nn
# import os
import numpy as np
# import math
import os
from torch.autograd import Variable
# from torch.utils.data import Dataset
# import torch.optim as optim 
# import random

# from sklearn.metrics import r2_score, explained_variance_score

# from networks.Network1_v1 import Network1, Network1_Data_loader
from networks.Network1_v5 import Network1, Network1_Data_loader
# from networks.Network1_v2 import Network1, Network1_Data_loader
# from networks.Network2_v2 import Network2, Network2_Data_loader
# from networks.Network3 import Network3, Network3_Data_loader

from matplotlib import pyplot as plt


def data_extractor(filename, processed_filename1, processed_filename2=None):
    # cleans IMU outputs and generates usable dataset as a clean CSV file

    file = open(filename, 'r')
    out_file1 = open(processed_filename1, 'w')
    if processed_filename2 is not None:
        out_file2 = open(processed_filename2, 'w')

    start_time, end_time, time  = '', '', ''
    lin_ctr = 0

    for l in file:
        if l[0] == '[':                              # if data has timestamp logging on
            time, line = l[10:22], l[24:].rstrip()
        else:
            line = l.rstrip()

        tab_ctr = line.count("\t")
        if tab_ctr == 3 or tab_ctr == 9:
            lin_ctr += 1
            line_lst = line.rstrip().split("\t")
#             if line_lst[0]=='LSM_raw: ' :
#                 print(line.rstrip(), file=out_file1, end=";\t\t")
            if line_lst[0]=='MPU_raw: ' :
                print(line.rstrip(), file=out_file1, end=";\t\t")
#             elif line_lst[0]=='LSM_ypr: ' :
#                 print(line.rstrip(), file=out_file1, end="\n")
            elif line_lst[0]=='MPU_ypr: ' :
                print(line.rstrip(), file=out_file1, end="\n")
#                 print(line.rstrip(), file=out_file2, end="\n")
            if lin_ctr == 1 :
                start_time = time
            

    end_time = time
    if end_time != '':
        print("Start_time = {}; End_time = {}".format(start_time, end_time), file=out_file1, end="\n")

    if (processed_filename2 is not None) and (end_time != ''):
        print("Start_time = {}; End_time = {}".format(start_time, end_time), file=out_file2, end="\n")
        out_file2.close()

    out_file1.close()
    file.close()
    
def file_train_test_splitter(input_file_name, train_percent):
    # splits the input file into two files given by the train_percent
        # example usage : file_train_test_splitter("input.txt", 70)
    
    with open(input_file_name, 'r') as f:
        for i, l in enumerate(f):
            pass
    lines_in_file = i + 1   # total number of lines, aka, data points in the input file
    f.close()
    
    try:
        ratio = train_percent / 100
    except:
        print("Error: Invalid train test split ratio")
    
    num_train_data_points = int(lines_in_file * ratio)
    num_test_data_points = lines_in_file - num_train_data_points
    
    train_out_file = open(os.path.splitext(input_file_name)[0]+"_train_data.csv", 'w')
    test_out_file = open(os.path.splitext(input_file_name)[0]+"_test_data.csv", 'w')
    
    line_ctr = 0
    with open(input_file_name, 'r') as f:
        for line in f:
            line_ctr += 1
            line = line.rstrip()
            if line_ctr <= num_train_data_points:
                print(line, file = train_out_file)
            else:
                print(line, file = test_out_file)
    
    f.close()
    test_out_file.close()
    train_out_file.close()
    
    print("Train and test files created.")


if __name__ == "__main__":
    
    # Data Pre-processing 
    data_extractor("Datasets/MPU_rawtoypr_11.csv", "Datasets/MPU_rawtoypr_11_cleaned.csv")
    file_train_test_splitter("Datasets/MPU_rawtoypr_11_cleaned.csv", 70)
    
    model = Network1(init_weights=True)
    print(model)
    
    # Initialize model weights from older checkpoint (if necessary)
    state_dict = model.load_from_checkpoint("/the_good_weights/net1_v5_network_state_checkpoint80.pth")
    model.load_state_dict(state_dict)
    
    data_train = Network1_Data_loader("Datasets/MPU_rawtoypr_11_cleaned_train_data.csv")    # training data LSM raw -> LSM ypr
    data_test = Network1_Data_loader("Datasets/MPU_rawtoypr_11_cleaned_test_data.csv") 
    

    print("Training Data matrix shape = ",data_train.data_matrix.shape)
    print("Test Data matrix shape = ",data_test.data_matrix.shape)
    
    # Initializing the DataLoader(s)
    dsets_enqueuer_training = torch.utils.data.DataLoader(data_train, batch_size=1, 
                                                          num_workers=1, drop_last=False)
    dsets_enqueuer_testing = torch.utils.data.DataLoader(data_test, batch_size=1, 
                                                          num_workers=1, drop_last=False)

    epochs = 50

    # Training 
    loss_lst_train, loss_lst_test, r_sq_lst, r_sq_lst_test,  var_exp_lst, var_exp_lst_test = model.train_model(epochs, 
                                                                                               dsets_enqueuer_training, 
                                                                                               dsets_enqueuer_testing)
    
    
    
    #Display Loss Graphs
    train_loss = np.squeeze(np.mean(np.array(loss_lst_train), axis=2))
    test_loss = np.squeeze(np.mean(np.array(loss_lst_test), axis=2))

    plt.close()
    plt.figure(num=None, figsize=(18, 24), dpi=80)

    plt.subplot(311)
    plt.title('Cosine Distance Error per Epoch', fontsize=20)
    plt.plot( train_loss , 'r')            # this is the mean squared error among yaw, pitch, roll angles
    plt.plot( test_loss  , 'g')            # can have a max value of 360^2 = 126000
    # plt.plot( np.array(loss_lst_train) , 'r')            # this is the mean squared error among yaw, pitch, roll angles
    # plt.plot( np.array(loss_lst_test)  , 'g')
    plt.ylabel("Cosine distance", fontsize=16)
    plt.xlabel("Epoch", fontsize=16)
    plt.legend(['training MSE', 'testing MSE'], loc='upper right')
    plt.grid()

    plt.subplot(312)
    plt.title("R squared Value", fontsize=20)
    plt.plot(r_sq_lst, 'r')
    plt.plot(r_sq_lst_test, 'g')
    plt.ylabel("R squared score", fontsize=16)
    plt.xlabel("Epoch", fontsize=16)
    plt.legend(['training R sq', 'testing R sq'], loc='upper right')
    plt.grid()

    plt.subplot(313)
    plt.title("Variance Explained", fontsize=20)
    plt.plot(np.array(var_exp_lst) * 100, 'r')
    plt.plot(np.array(var_exp_lst_test) * 100, 'g')
    plt.ylabel("Explained Variance (%)", fontsize=16)
    # plt.ylim(-100,110)
    plt.xlabel("Epoch", fontsize=16)
    plt.legend(['training Var exp', 'testing Var exp'], loc='upper right')
    plt.grid()

    plt.savefig("saved_model_weights/Net1_v5/loss_graphs.png")
    #plt.show()
   