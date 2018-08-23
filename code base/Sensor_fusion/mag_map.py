import torch
import torch.nn as nn
# import os
import numpy as np
# import math
import os
from torch.autograd import Variable
from torch.utils.data import Dataset
# import torch.optim as optim 
    # import random1

from networks.mag_mapper import Mag_mapper, Mag_mapper_Data_loader

from matplotlib import pyplot as plt


# Data Preprocessing

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
            if line_lst[0]=='LSM_raw: ' :
                print(line.rstrip(), file=out_file1, end=";\t\t")
            elif line_lst[0]=='MPU_raw: ' :
                print(line.rstrip(), file=out_file1, end="\n")
            #elif line_lst[0]=='MPU_ypr: ' :
            #    print(line.rstrip(), file=out_file1, end="\n")
            #    print(line.rstrip(), file=out_file2, end="\n")
            if lin_ctr == 1 :
                start_time = time
            #elif line_lst[0]=='LSM_ypr: ' :
            #    print(line.rstrip(), file=out_file1, end="\n")
    
    end_time = time
    print("Start_time = {}; End_time = {}".format(start_time, end_time), file=out_file1, end="\n")
    
    if processed_filename2 is not None:
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

    data_extractor("Datasets/New_train_test_datasets/imu_mag_map_dataset1.txt", 
                   "Datasets/New_train_test_datasets/imu_mag_map_dataset1_cleaned.txt")

    file_train_test_splitter("Datasets/New_train_test_datasets/imu_mag_map_dataset1_cleaned.txt", 70)

    for i in range(10,151,10):
        print("----------------------------------------------------------------------------------")
        model = Mag_mapper(init_weights=True, hidden=i)   # i = 10,20,30, ... ,130,140,150
        print(model)

        # training data LSM raw -> MPU ypr
        data_train = Mag_mapper_Data_loader("Datasets/New_train_test_datasets/imu_mag_map_dataset1_cleaned_train_data.csv")    
        data_test = Mag_mapper_Data_loader("Datasets/New_train_test_datasets/imu_mag_map_dataset1_cleaned_test_data.csv")

        print("Total training data samples = ", len(data_train))
        print("Training Data matrix shape = ",data_train.data_matrix.shape)

        print("Total test data samples = ", len(data_test))
        print("Test Data matrix shape = ",data_test.data_matrix.shape)

        # Initializing the DataLoader(s)
        dsets_enqueuer_training = torch.utils.data.DataLoader(data_train, batch_size=1, 
                                                              num_workers=1, drop_last=False)
        dsets_enqueuer_testing = torch.utils.data.DataLoader(data_test, batch_size=1, 
                                                              num_workers=1, drop_last=False)

        epochs = 250

        # Training
        loss_lst_train, loss_lst_test, r_sq_lst, r_sq_lst_test,  var_exp_lst, var_exp_lst_test = model.train_model(epochs, 
                                                                                                       dsets_enqueuer_training, 
                                                                                                       dsets_enqueuer_testing)

        train_loss = np.squeeze(np.mean(np.array(loss_lst_train), axis=1))
        test_loss = np.squeeze(np.mean(np.array(loss_lst_test), axis=1))

        plt.close()
        plt.figure(num=None, figsize=(18, 20), dpi=80)

        plt.subplot(311)
        plt.title('MSE per Epoch', fontsize=20)
        plt.plot( train_loss , 'r')           
        plt.plot( test_loss  , 'g')            
        # plt.plot( np.array(loss_lst_train) , 'r')            
        # plt.plot( np.array(loss_lst_test)  , 'g')
        plt.ylabel("MSE", fontsize=16)
        plt.xlabel("Epoch", fontsize=13)
        # plt.ylim(-0.025,0.225)
        plt.legend(['training MSE', 'testing MSE'], loc='upper right')
        plt.grid()

        plt.subplot(312)
        plt.title("R squared Value", fontsize=20)
        plt.plot(r_sq_lst, 'r')
        plt.plot(r_sq_lst_test, 'g')
        plt.ylabel("R squared score", fontsize=16)
        plt.xlabel("Epoch", fontsize=13)
        plt.legend(['training R sq', 'testing R sq'], loc='upper right')
        plt.grid()

        plt.subplot(313)
        plt.title("Variance Explained", fontsize=20)
        plt.plot(np.array(var_exp_lst) * 100, 'r')
        plt.plot(np.array(var_exp_lst_test) * 100, 'g')
        plt.ylabel("Explained Variance (%)", fontsize=16)
        # plt.ylim(-50,110)
        plt.xlabel("Epoch", fontsize=13)
        plt.legend(['training Var exp', 'testing Var exp'], loc='upper right')
        plt.grid()

        plt.savefig("mag_map_network_training_loss_h{}.png".format(i))
        #plt.show()