import numpy as np
import io
import os
import matplotlib.pyplot as plt
import scipy.io

data_path = "/lab/mksimmon/Downloads/Shixian_Transformer/Dataset/data_prepro"
num_snips = 71
num_features = 48

dir_list_data = os.listdir(data_path)
dir_list_data = sorted(dir_list_data)

max_vector_length = 0


for file in dir_list_data:
    array = np.load(f"{data_path}/{file}", allow_pickle = True)
    time = array[0][0].shape[0]
    if time > max_vector_length:
        max_vector_length = time
        max_file = file
    
max_vector_length = max_vector_length + 1 #making it divisible by 2

for file in dir_list_data:
    padded_array = np.load(f"{data_path}/{file}", allow_pickle = True)
    num_people = padded_array.shape[0]
    for person in range(num_people):
        #print(person)
        #print(max_vector_length - padded_array[person][0].shape[0])
        if (max_vector_length - padded_array[person][0].shape[0]) != max_vector_length: 
            updated_padded_array = np.pad(padded_array[person][0], ((0, (max_vector_length - padded_array[person][0].shape[0])), (0, 0)), mode='constant', constant_values = 0)
            padded_array[person][0] = updated_padded_array
        else:
            #print(f"person {person} did not have a vector for this snip, so I added one")
            vector = np.zeros((max_vector_length, num_features)) #vector of zeros that we're adding as a placeholder for even dimensions
            reshaped = np.reshape(padded_array[person][0], (0, num_features))
            updated_padded_array = np.concatenate((reshaped, vector), axis=0)
            padded_array[person][0] = updated_padded_array
    path = os.path.join(data_path, file)

    np.save(path, padded_array) #saving all the padded bois
