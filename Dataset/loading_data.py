import torch
import tarfile
import numpy as np
import io
import os
import matplotlib.pyplot as plt

data_path = "/lab/mksimmon/Downloads/Shixian_Transformer/Dataset/data_prepro"
target_path = "/lab/mksimmon/Downloads/Shixian_Transformer/Dataset/target"
folder_path = '/lab/raid/datasets/eye-tracking/iLab-Queens-FASD-EyeTracking/FASD eye dataset'

num_snips = 70

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
