import numpy as np
import io
import os
import matplotlib.pyplot as plt
import scipy.io

data_path = "/lab/mksimmon/Downloads/Shixian_Transformer/Dataset/data_prepro"
padded_path = "/lab/mksimmon/Downloads/Shixian_Transformer/Dataset/data_padded"
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


print(max_vector_length)

correct_length = 3984

print(max_vector_length)
# padded_array = np.load(f"{data_path}/{dir_list_data[0]}", allow_pickle = True)

# print(f"shape before removing: {padded_array[1][0].shape}")
# new_padded = np.delete(padded_array[1][0], np.s_[3984:max_vector_length], axis = 0) # Deleting the extra padded entries I added by accident
# padded_array[1][0] = new_padded

# print(f"shape after deletion: {padded_array[1][0].shape}")

for file in dir_list_data:
    padded_array = np.load(f"{data_path}/{file}", allow_pickle = True)
    num_people = padded_array.shape[0]
    for person in range(num_people):
        new_padded = np.delete(padded_array[person][0], np.s_[correct_length:max_vector_length], axis = 0) # Deleting the extra padded entries I added by accident
        padded_array[person][0] = new_padded
    
    padded_data_path = os.path.join(padded_path, file)
    np.save(padded_data_path, padded_array) # Saving the new padded files
    print(f"saved file {file} in data path")



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

    np.save(padded_path, padded_array) #saving all the padded bois
