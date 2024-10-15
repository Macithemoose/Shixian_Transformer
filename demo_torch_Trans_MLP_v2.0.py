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
path_data = "/lab/mksimmon/Downloads/Shixian_Transformer/Dataset/data_padded"
dir_list_data = os.listdir(path_data)
dir_list_data = sorted(dir_list_data)

path_target = "/lab/mksimmon/Downloads/Shixian_Transformer/Dataset/target"
dir_list_target = os.listdir(path_target)
dir_list_target = sorted(dir_list_target)
print('All data and GT are matched: ', all([a[0:-4] == b[0:-4] for a, b in zip(dir_list_data, dir_list_target)]))


        
########################################################################################################################################################################            
################################################################### Training ###########################################################################################       
N = len(dir_list_data) # batch size
T = 3984 # Duration of data in frame
D = 48 # Degree of features - 48 features
Compact_L = 1024
epochs = 50
files_to_pick = 2 # For randomly picking snips to create batches
people_to_pick = 4 # For randomly picking people's data from the files.
batch_size = 8 # Require to be dividable by number of classes
slide_N = 71 # Number of slide windows
prepro_N = 20  # Number of sub-stack before embedding to transformers
Overlap_rate = 0 # Overlapping ratio of slide windows
input_shape = np.array([int(T), int(D)]) 
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
classifier_Transformer_model = nn.DataParallel(classifier_Transformer_model, device_ids=[0, 1])
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

    for i in tqdm(range(int(np.floor(L/batch_size*n_class)))): 
        count = count + 1   
        # sample_temp = np.zeros((batch_size,int(T/2),D))
        sample_temp = np.zeros((batch_size,T,D))
        target_temp = np.zeros((batch_size,n_class))
        index_batch_0 = index_0_train[np.random.choice(len(index_0_train))]
        index_batch_1 = index_1_train[np.random.choice(len(index_1_train))]
        index_batch = np.concatenate((index_batch_0,index_batch_1),axis=0)
        random_people = np.random.choice(47, 8, replace = False)
        sample_data = np.load(f"{path_data}/{dir_list_data[index_batch[0]]}", allow_pickle = True)
        print(f"Shape of sample data: {sample_data.shape}")
        first_person = sample_data[random_people[0]]
        result = np.array([obj for obj in first_person])
        print(result.shape)
        sample_temp[0] = np.expand_dims(result, 0)
        target_temp[0] = np.expand_dims(loadmat(path_target+'/'+str(dir_list_target[index_batch[0]]))['target'], 0)
        print(f"Sample temp shape: {sample_temp.shape}")
       
        for k in range(1, batch_size): # load in batchsize
            # print("The current data path is ", path_data+'/'+str(dir_list_data[index_batch[k+1]]))
            sample_data_control = np.load(f"{path_data}/{dir_list_data[index_batch[0]]}", allow_pickle = True)
            sample_data_fasd = np.load(f"{path_data}/{dir_list_data[index_batch[1]]}", allow_pickle = True)
            if k < batch_size/n_class:
                sample_obj = sample_data_control[random_people[k]]
                result = np.array([obj for obj in sample_obj])
                sample_temp[k] = np.expand_dims(result, 0)
                target_temp[k] = np.expand_dims(loadmat(path_target+'/'+str(dir_list_target[index_batch[0]]))['target'], 0)
            if k >= btach_size/n_class:
                sample_obj = sample_data_fasd[random_people[k]]
                result = np.array([obj for obj in sample_obj])
                sample_temp[k] = np.expand_dims(result, 0)
                target_temp[k] = np.expand_dims(loadmat(path_target+'/'+str(dir_list_target[index_batch[1]]))['target'], 0)
        target_temp = np.expand_dims(target_temp,2)
        target_temp = np.repeat(target_temp,slide_N+1,axis=2) 
        sample_total = torch.Tensor(sample_temp)
        target_total = torch.Tensor(target_temp)
        if cuda:
            sample_total = sample_total.cuda()
            target_total = target_total.cuda()   

        Pred_minor, Pred_major = classifier_Transformer_model(sample_total)
        #print("the shape of Pred_minor is",Pred_minor.shape)



        #print("the shape of Pred_major is",Pred_major.shape)
        
        #print("target_total[:,:,0:slide_N]",target_total[:,:,0:slide_N].shape)

        #print("the Pred_major is",Pred_major)
        #print("Pred_minor is",Pred_minor)
        #print("target_total is",target_total)
        loss1 = loss_function(Pred_major, target_total[:,:,slide_N]) 
        loss2 = loss_function(Pred_minor, target_total[:,:,0:slide_N])
        loss = (1-Lambda)*loss1 + Lambda*loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = torch.sum(torch.argmax(Pred_major, 1) == torch.argmax(target_total[:,:,slide_N], 1))/target_total.shape[0]
        #if np.mod(L,batch_size/n_class) == 0:
            #print(Fore.GREEN+'Epoch: ',j+1, '/',int(epochs),' Iteration: ', i+1, '/', int(np.floor(L/batch_size*n_class)),', Loss_major: ',"{:.4f}".format(loss1.item()),', Loss_minor ',"{:.4f}".format(loss2.cpu().data.item()),', Accuracy: ',"{:.2f}".format(acc.item()*100),'%')  
        #else:
            #print(Fore.GREEN+'Epoch: ',j+1, '/',int(epochs),' Iteration: ', i+1, '/', int(np.floor(L/batch_size*n_class))+1,', Loss_major: ',"{:.4f}".format(loss1.item()),', Loss_minor ',"{:.4f}".format(loss2.cpu().data.item()),', Accuracy: ',"{:.2f}".format(acc.item()*100),'%') 

        train_loss.append(loss.cpu().data.item())
        train_acc.append(acc.cpu().data.item())
        epoch_loss = epoch_loss + loss.cpu().data.item()
        epoch_acc = epoch_acc + acc.cpu().data.item()
        TP = TP + (torch.mul(torch.argmax(Pred_major, 1),target_total[:,1,slide_N])).sum().item()            
        pos = pos + torch.argmax(Pred_major, 1).sum().item()
        
        del sample_temp
        del target_temp
        del sample_total
        del target_total
        #print(Fore.GREEN+'#################### The CPU usage is: ', psutil.virtual_memory()[3]/1000000000)
        #torch.cuda.empty_cache()
        #print(Fore.GREEN+'--- ',"{:.4f}".format((time.time() - start_time)/60),' minutes escaped ---')
        #print()

    if np.mod(L,batch_size/n_class) != 0: ### the last iteration if not be a full batch
        count = count + 1
        index_batch_0 = np.concatenate((index_0_train[-int(np.mod(L,batch_size/n_class)): -1],np.expand_dims(index_0_train[-1],axis=0)),axis=0)
        index_batch_1 = np.concatenate((index_1_train[-int(np.mod(L,batch_size/n_class)): -1],np.expand_dims(index_1_train[-1],axis=0)),axis=0)
        index_batch = np.concatenate((index_batch_0,index_batch_1),axis=0)
        index_batch = index_batch[np.random.permutation(int(np.mod(L,batch_size/n_class)*n_class))]
        sample_temp = np.zeros((int(np.mod(L,batch_size/n_class)*n_class),int(T/2),D))
        target_temp = np.zeros((int(np.mod(L,batch_size/n_class)*n_class),n_class))
        sample_temp[0] = np.expand_dims(loadmat(path_data+'/'+str(dir_list_data[index_batch[0]]))['data'],0)
        target_temp[0] = np.expand_dims(loadmat(path_target+'/'+str(dir_list_target[index_batch[0]]))['target'],0)
        
        for k in range(int(np.mod(L,batch_size/n_class)*n_class-1)): # load in batchsize
            sample_temp[k+1] = np.expand_dims(loadmat(path_data+'/'+str(dir_list_data[index_batch[k+1]]))['data'],0)
            target_temp[k+1] = np.expand_dims(loadmat(path_target+'/'+str(dir_list_target[index_batch[k+1]]))['target'],0)

        target_temp = np.expand_dims(target_temp,2)
        target_temp = np.repeat(target_temp,slide_N+1,axis=2) 
        sample_total = torch.Tensor(sample_temp)
        target_total = torch.Tensor(target_temp)
        if cuda:
            sample_total = sample_total.cuda()
            target_total = target_total.cuda()    

        Pred_minor, Pred_major = classifier_Transformer_model(sample_total)
        loss1 = loss_function(Pred_major[0:int(np.mod(L,batch_size/n_class)*n_class)], target_total[:,:,slide_N]) 
        loss2 = loss_function(Pred_minor[0:int(np.mod(L,batch_size/n_class)*n_class)], target_total[:,:,0:slide_N])
        loss = (1-Lambda)*loss1 + Lambda*loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = torch.sum(torch.argmax(Pred_major[0:int(np.mod(L,batch_size/n_class)*n_class)], 1) == torch.argmax(target_total[:,:,slide_N], 1))/target_total.shape[0]
        #print(Fore.GREEN+'Epoch: ',j+1, '/',int(epochs),' Iteration: ', i+2, '/', int(np.floor(L/batch_size*n_class))+1,', Loss_major: ',"{:.4f}".format(loss1.item()),', Loss_minor ',"{:.4f}".format(loss2.item()),', Accuracy: ',"{:.2f}".format(acc.item()*100),'%')  

        train_loss.append(loss.cpu().data.item())
        train_acc.append(acc.cpu().data.item())
        epoch_loss = epoch_loss + loss.cpu().data.item()
        epoch_acc = epoch_acc + acc.cpu().data.item()
        TP = TP + (torch.mul(torch.argmax(Pred_major[0:int(np.mod(N,batch_size))], 1), target_total[0:int(np.mod(N,batch_size)),1,slide_N])).sum().item()            
        pos = pos + torch.argmax(Pred_major[0:int(np.mod(N,batch_size))], 1).sum().item()
        del sample_temp
        del target_temp
        del sample_total
        del target_total
        #print(Fore.GREEN+'#################### The CPU usage is: ', psutil.virtual_memory()[3]/1000000000)
        #print(Fore.GREEN+'--- ',"{:.4f}".format((time.time() - start_time)/60),' minutes escaped ---')
        #print()
    
    print()
    print('The current epoch: ',j+1, '/',int(epochs), 'average training loss: ',"{:.4f}".format(epoch_loss/count), ' training acc: ', "{:.4f}".format(epoch_acc*100/count),'%')
    print('Precision: ',"{:.2f}".format((TP+1e-6)/(pos+1e-6)),' ---') 
    print('Recall: ',"{:.2f}".format(TP/L),' ---')
    ### Evaluate model
    with torch.no_grad():
        classifier_Transformer_model.eval()
        loss = 0
        acc = 0
        N = len(val_0)+len(val_1)
        index_val = np.concatenate((val_0,val_1),axis=0)
        index_val = index_val[np.random.permutation(int(N))]
        TT = 0
        pos = 0
        TP = 0
        for i in range(int(np.floor(N/batch_size))):
            TT += 1
            index_batch = index_val[i*int(batch_size):(i+1)*int(batch_size)]
            sample_temp = np.zeros((batch_size,int(T/2),D))
            target_temp = np.zeros((batch_size,n_class))
            sample_temp[0] = np.expand_dims(loadmat(path_data+'/'+str(dir_list_data[index_batch[0]]))['data'],0)
            target_temp[0] = np.expand_dims(loadmat(path_target+'/'+str(dir_list_target[index_batch[0]]))['target'],0)
            for k in range(batch_size-1): # load in batchsize
                sample_temp[k+1] = np.expand_dims(loadmat(path_data+'/'+str(dir_list_data[index_batch[k+1]]))['data'],0)
                target_temp[k+1] = np.expand_dims(loadmat(path_target+'/'+str(dir_list_target[index_batch[k+1]]))['target'],0)

            sample_total = torch.Tensor(sample_temp)
            target_total = torch.Tensor(target_temp)
            if cuda:
                sample_total = sample_total.cuda()
                target_total = target_total.cuda()  
            _, Predictions = classifier_Transformer_model(sample_total)

            # You need transfer it to .item() to stop memory explotion
            loss = loss + loss_function(Predictions, target_total).item()
            acc = acc + (torch.argmax(Predictions, 1) == torch.argmax(target_total, 1)).sum().item()            
            TP = TP + (torch.mul(torch.argmax(Predictions, 1),target_total[:, 1])).sum().item()            
            pos = pos + torch.argmax(Predictions, 1).sum().item()

            del sample_temp
            del target_temp
            del sample_total
            del target_total

        if np.mod(N,batch_size) != 0: ### the last iteration if not be a full batch
            TT += 1
            index_batch = np.concatenate((index_val[-int(np.mod(N,batch_size)): -1],np.expand_dims(index_val[-1],axis=0)),axis=0)
            sample_temp = np.zeros((int(np.mod(N,batch_size)),int(T/2),D))
            target_temp = np.zeros((int(np.mod(N,batch_size)),n_class))
            sample_temp[0] = np.expand_dims(loadmat(path_data+'/'+str(dir_list_data[index_batch[0]]))['data'],0)
            target_temp[0] = np.expand_dims(loadmat(path_target+'/'+str(dir_list_target[index_batch[0]]))['target'],0)
            
            for k in range(int(np.mod(N,batch_size)-1)): # load in batchsize
                sample_temp[k+1] = np.expand_dims(loadmat(path_data+'/'+str(dir_list_data[index_batch[k+1]]))['data'],0)
                target_temp[k+1] = np.expand_dims(loadmat(path_target+'/'+str(dir_list_target[index_batch[k+1]]))['target'],0)
            sample_total = torch.Tensor(sample_temp)
            target_total = torch.Tensor(target_temp)
            if cuda:
                sample_total = sample_total.cuda()
                target_total = target_total.cuda()    

            _, Predictions = classifier_Transformer_model(sample_total)
            loss = loss + loss_function(Predictions[0:int(np.mod(N,batch_size))], target_total).item()
            acc = acc + (torch.argmax(Predictions[0:int(np.mod(N,batch_size))], 1) == torch.argmax(target_total, 1)).sum().item()
            TP = TP + (torch.mul(torch.argmax(Predictions[0:int(np.mod(N,batch_size))], 1), target_total[:, 1])).sum().item()            
            pos = pos + torch.argmax(Predictions, 1).sum().item()

            del sample_temp
            del target_temp
            del sample_total
            del target_total
        loss = loss/TT
        acc = acc/N

        print('Loss: ',"{:.4f}".format(loss),', Validation accuracy of ',N,' samples: ',"{:.2f}".format(acc*100),'% ---\x1b[0m') 
        #print('\x1b[31m --- Precision: ',"{:.2f}".format((TP+1e-6)/(pos+1e-6)),' ---\x1b[0m') 
        #print('\x1b[31m --- Recall: ',"{:.2f}".format(TP/len(val_1)),' ---\x1b[0m') 
        print('\x1b[32m --- ',"{:.4f}".format((time.time() - start_time)/60),' minutes escaped ---\x1b[0m')
        eval_loss.append(loss)
        eval_acc.append(acc)

        my_lr_scheduler.step()
        
        #torch.cuda.empty_cache()
        print('\x1b[32m************************** The CPU usage is: ', psutil.virtual_memory()[3]/1000000000,'**********************\x1b[0m')
        #save best model here
        if(acc_val_best<acc):
            acc_val_best = acc
            pre_val_best = (TP+1e-6)/(pos+1e-6)
            rec_val_best = TP/len(val_1)
             
            
            save_on_master({'model_state_dict': classifier_Transformer_model.state_dict()}, './Save/'+'Classifier_T_batchsize_'+str(batch_size)+'_epochs_'+str(epochs)+'_slide_N_'+str(slide_N)+'_prepro_N_'+str(prepro_N) \
                              +'_Overlap_rate_'+str(Overlap_rate)+'_num_heads_'+str(num_heads)+'_headsize_'+str(headsize)+'_ff_dim_'+str(ff_dim) \
                                  +'_num_transformer_block_'+str(num_transformer_block)+'_n_class_'+str(n_class)+'_Weight_'+str(Lambda)+'sub_factor'+str(sub_factor)+'.pth')  
        print('\x1b[32m --- Best Accuracy: ',"{:.2f}".format(acc_val_best*100),'%  ---\x1b[0m')             
        print('\x1b[32m --- Precision: ',"{:.2f}".format(pre_val_best),' ---\x1b[0m') 
        print('\x1b[32m --- Recall: ',"{:.2f}".format(rec_val_best),' ---\x1b[0m')
        print()
        del loss
        del acc

savemat('train_loss.mat',{'train_loss': np.array(train_loss)})
savemat('train_acc.mat',{'train_acc': np.array(train_acc)})
savemat('eval_loss.mat',{'eval_loss': np.array(eval_loss)})
savemat('eval_acc.mat',{'eval_acc': np.array(eval_acc)})
               
    
    


     
     
            
            
            


'''
k_folds = 5
num_epochs = 100

# For fold results
results = {}
# Set fixed random number seed
torch.manual_seed(42)
# Define the K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True)
# K-fold Cross Validation model evaluation
for fold, (train_ids, test_ids) in enumerate(kfold.split(data_set)):    
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
      
    # Define data loaders for training and testing data in this fold
    train_data_loader = torch.utils.data.DataLoader(data_set,batch_size=3, sampler=train_subsampler)
    train_target_loader = torch.utils.data.DataLoader(target_set,batch_size=3, sampler=train_subsampler)
    test_data_loader = torch.utils.data.DataLoader(data_set,batch_size=3, sampler=test_subsampler)
    test_target_loader = torch.utils.data.DataLoader(target_set,batch_size=3, sampler=test_subsampler)
    #####################################################################################################################
    #Not sure about this section, previously we already load all the data_set onto the GPU
    #What we need to change is, load a batch_size =16 data only on the GPU
    #Every time we fetch a new batch, we read from the disk and load them to the GPU
    #Here I need to see the overlap implementation on the original data, maybe apply a for loop and apply a slide window as an input the network
    #####################################################################################################################

  
    # Initialize optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
  
    # Run the training loop for defined number of epochs
    for epoch in range(0, num_epochs):
  #####################################################################################################################
    #in the training phase, tell the network to enable .train(), activate dropout .... etc
    #####################################################################################################################

        network.train()
        # Print epoch
        print(f'Starting epoch {epoch+1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for data, target in zip((enumerate(train_data_loader, 0)), enumerate(train_target_loader, 0)):
            # Get inputs
            if cuda:
                data = data.cuda()
                target= target.cuda()
    #####################################################################################################################
    #at here, load the date on to the gpu.
    #####################################################################################################################





            i = data[0]
            inputs = data[1]
            #inputs = torch.transpose(inputs, 1, 2)
            targets = target[1]
            
      
            # Zero the gradients
            optimizer.zero_grad()
      
            # Perform forward pass
            outputs = network(inputs)
      
            # Compute loss
            loss = loss_function(outputs, targets)
      
            # Perform backward pass
            loss.backward()
      
            # Perform optimization
            optimizer.step()
      
            # Print statistics
            current_loss += loss.item()
            if i % 10 == 9: #we may increase this after getting more samples
                print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss/10.0))
                current_loss = 0.0
          
    # Process is complete.
    print('Training process has finished. Saving trained model.')

    # Print about testing
    print('Starting testing')
  
    # Saving the model
    save_path = f'./model-fold-{fold}.pth'
    torch.save(network.state_dict(), save_path)

    # Evaluationfor this fold
    correct, total = 0, 0

    #####################################################################################################################
    #Don't do torch.no_grad():
    # Just do three step:
    # 1: network.eval()  tell the network it is in the evaluation phase
    # 2: out = network(input)
    # 3: calculate error rate,  
    # In addition, the test phase should be inside of the training epochs
    # maybe check them every 5 epochs, save the best model on test sets on the diskss
    # maybe try one fold first, them apply to 5 folds, don't have to so hurry now.
    #####################################################################################################################

    with torch.no_grad():
        # Iterate over the test data and generate predictions
        for data, target in zip(enumerate(test_data_loader, 0),enumerate(test_target_loader,0)):
            # Get inputs
            i = data[0]
            inputs = data[1]
            #inputs = torch.transpose(inputs, 1, 2)
            targets = target[1]
            # Generate outputs
            outputs = network(inputs)

            # Set total and correct
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # Print accuracy
        print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
        print('--------------------------------')
        results[fold] = 100.0 * (correct / total)
  
# Print fold results
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
print('--------------------------------')
sum = 0.0
for key, value in results.items():
    print(f'Fold {key}: {value} %')
    sum += value
print(f'Average: {sum/len(results.items())} %')
print('--- ',"{:.4f}".format((time.time() - start_time)/60),' minutes escaped (finished) ---')
'''
