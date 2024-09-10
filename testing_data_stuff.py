#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 12:55:49 2024

@author: Kangning Zhang and Shixian Wen
"""
import numpy as np
import os
import scipy
import time
import torch
from torch import nn
from scipy.io import loadmat, savemat
from tqdm import tqdm
#from torchsummary import summary
#from sklearn.model_selection import KFold
import psutil
#from colorama import Fore
from networks import Transformers_Encoder_Classifier, save_on_master


    
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(torch.__version__)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor 

start_time = time.time()
#################################################################################################################################################################
################################################# Classifier based on transformers and MLP ######################################################################

######################################################################################################################################################################
###################################################### Data Loading and Processing ###################################################################################
# Define the path, load all the file names in the path, and sort the names
path_data = "/lab/mksimmon/Downloads/Shixian_Transformer/Dataset/data_prepro"
dir_list_data = os.listdir(path_data)
dir_list_data = sorted(dir_list_data)

path_target = "/lab/mksimmon/Downloads/Shixian_Transformer/Dataset/target"
dir_list_target = os.listdir(path_target)
dir_list_target = sorted(dir_list_target)
print('All data and GT are matched: ', all([a[0:-4] == b[0:-4] for a, b in zip(dir_list_data, dir_list_target)]))


        
########################################################################################################################################################################            
################################################################### Training ###########################################################################################       
N = len(dir_list_data) # batch size
T = 1024 # Duration of data in frame
D = 48 # Degree of features - 48 features, must be divisible by th number of heads
Compact_L = 1024
epochs = 400
batch_size = 8 # Require to be dividable by number of classes
slide_N = 71 # Number of slide windows
prepro_N = 20  # Number of sub-stack before embedding to transformers
Overlap_rate = 0 # Overlapping ratio of slide windows
input_shape = np.array([int(T/2), int(D)]) 
num_heads = 4 # number of heads in multihead attention layer
headsize = 128 # degree of the variable tensor in self-attention
ff_dim = 512  # degree of the variable in feed forward layer
num_transformer_block = 6 # number of transformer encoder blocks
mlp_units = [512, 16]  # number of channels in MLP before classification
drop = 0.4 
mlp_drop = 0.4
n_class = 2
flag = 0 # Show input and output shape
d_model = input_shape[1]
#d_model = 128
self_attn_numhead = 4
train_ratio = 0.8 # The ratio of training set in the whole data
learning_rate = 5*1e-4
decayRate = 0.98
Lambda = 0.2 # The weight of minority part of the loss. Loss = (1-lambda)*Loss_major(final_pred, target) + lambda*Loss_minor(slide_pred, target)
sub_factor = 8
index = np.zeros((len(dir_list_data),n_class))
for i in range(int(len(dir_list_data))):
    target = scipy.io.loadmat(path_target+'/'+str(dir_list_target[i]))
    target = target['target']
    if (target == np.array([1,0])).all():
        index[i,0] = 1
    elif (target == np.array([0,1])).all():
        index[i,1] = 1
index_0 = np.where(index[:,0] == 1)[0]
index_0 = index_0[np.random.permutation(int(len(index_0)))]
index_1 = np.where(index[:,1] == 1)[0]
index_1 = index_1[np.random.permutation(int(len(index_1)))]

train_0 = index_0[0:int(len(index_0)*train_ratio)]
val_0 = index_0[int(len(index_0)*train_ratio):int(len(index_0))]
train_1 = index_1[0:int(len(index_1)*train_ratio)]
val_1 = index_1[int(len(index_1)*train_ratio):int(len(index_1))]


classifier_Transformer_model = Transformers_Encoder_Classifier(batch_size, slide_N, prepro_N, Overlap_rate, input_shape, num_heads, headsize, ff_dim, 
                                                               num_transformer_block, mlp_units, drop, mlp_drop, n_class,epochs,flag,
                                                               d_model,self_attn_numhead,sub_factor,Compact_L)
classifier_Transformer_model = nn.DataParallel(classifier_Transformer_model, device_ids=[0, 1]) #changed to only two devices  
if cuda:
    classifier_Transformer_model.cuda()

total_params = sum(p.numel() for p in classifier_Transformer_model.parameters())
print(f"Number of trainable parameters: {total_params}")
Weight=torch.Tensor(np.array([0.35,0.65]))
if cuda:
    Weight = Weight.cuda()

loss_function = nn.CrossEntropyLoss(weight=Weight)
optimizer = torch.optim.Adam(classifier_Transformer_model.parameters(), lr=learning_rate)
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
loss_history = {}
#results = {}
start_time = time.time()

train_loss = []
train_acc = []
eval_loss = []
eval_acc = []
acc_val_best = 0

for j in range(epochs):

    # print the average epoch loss and accuracy
    epoch_loss = 0;
    epoch_acc = 0;
    count = 0;
    pos = 0
    TP = 0
    
    ### Balancing the amount of each class
    L = np.maximum(int(len(train_0)),int(len(train_1)))
    index_0_train  = []
    index_1_train = []
    for k in range(int(np.floor(L/len(train_0)))):
        index_0_train = np.concatenate((index_0_train, train_0[np.random.permutation(int(len(train_0)))]),axis=0)
    index_0_train = np.concatenate((index_0_train,train_0[np.random.permutation(int(len(train_0)))[0:np.mod(L,len(train_0))]]),axis=0)
    
    for k in range(int(np.floor(L/len(train_1)))):
        index_1_train = np.concatenate((index_1_train, train_1[np.random.permutation(int(len(train_1)))]),axis=0)
    index_1_train = np.concatenate((index_1_train,index_1_train[np.random.permutation(int(len(train_1)))[0:np.mod(L,len(train_1))]]),axis=0) 
    index_0_train = np.intc(index_0_train)
    index_1_train = np.intc(index_1_train)
    
    classifier_Transformer_model.train()

    for i in tqdm(range(0, 1)): 
        count = count + 1   
        sample_temp = np.zeros((batch_size,int(T/2),D))
        target_temp = np.zeros((batch_size,n_class))
        index_batch_0 = index_0_train[i*int(batch_size/n_class):(i+1)*int(batch_size/n_class)]
        index_batch_1 = index_1_train[i*int(batch_size/n_class):(i+1)*int(batch_size/n_class)]
        index_batch = np.concatenate((index_batch_0,index_batch_1),axis=0)
        index_batch = index_batch[np.random.permutation(batch_size)]
        sample_data = np.load(f"{path_data}/{dir_list_data[index_batch[0]]}", allow_pickle = True)
        first_obj = sample_data[0]
        result = np.array([obj for obj in first_obj])
        print(result.shape)
        sample_temp[0] = np.expand_dims(result, 0)
        target_temp[0] = np.expand_dims(loadmat(path_target+'/'+str(dir_list_target[index_batch[0]]))['target'],0)
