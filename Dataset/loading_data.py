import torch
import tarfile
import numpy as np
import io
import os
import matplotlib.pyplot as plt
import scipy.io
import glob

data_path = "/lab/mksimmon/Downloads/Shixian_Transformer/Dataset/data_prepro"
target_path = "/lab/mksimmon/Downloads/Shixian_Transformer/Dataset/target"
folder_path = '/lab/raid/datasets/eye-tracking/iLab-Queens-FASD-EyeTracking/FASD eye dataset'

num_snips = 70

def load_snips(num_snips):
    for snip in range(1, num_snips + 1):
        filename = f'snip{snip}.tar.gz'
        tar_path = os.path.join(folder_path, filename)
        with tarfile.open(tar_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.name.endswith('fasd_rawdata.npy'):
                    npy_file = tar.extractfile(member)
                    path = os.path.join(data_path, f'fasd_snip{snip}.npy')
                    loaded_array = np.load(npy_file,allow_pickle=True)
                    np.save(path, loaded_array)
                if member.name.endswith('control_rawdata.npy'):
                    npy_file = tar.extractfile(member)
                    path = os.path.join(data_path, f'control_snip{snip}.npy')
                    loaded_array = np.load(npy_file,allow_pickle=True)
                    np.save(path, loaded_array)
                

def create_targets(num_snips, data_path, target_path):
    #format of each target: {"target": [1, 0]}
    dir_list_data = os.listdir(data_path)
    for file in range(1, (len(dir_list_data) + 1)): # since need a label for each control snip and each fasd snip
        if file <= num_snips:
            data = {'target': [1, 0]} #one-hot encoding for the control label, creating a .mat dictionary to be read by the model
            file_path = os.path.join(target_path, f'control_snip{file}.mat')
            scipy.io.savemat(file_path, data)
        if file > num_snips:
            data = {'target' : [0, 1]} #one-hot encoding for the fasd label, creating a .mat dictionary to be read by the model
            file_path = os.path.join(target_path, f'fasd_snip{file - num_snips}.mat') #starting the count over to match snips
            scipy.io.savemat(file_path, data)

def delete_incorrect(num_snips, target_path):
    search_string = 'target'
    files_to_delete = glob.glob(os.path.join(target_path, f'*{search_string}*'))
    for file in files_to_delete:
        try:
            os.remove(file)
        except Exception as e:
            print(f'Error deleting file {file}: {e}')

#load_snips(num_snips)
create_targets(num_snips, data_path, target_path)
#delete_incorrect(num_snips, target_path)