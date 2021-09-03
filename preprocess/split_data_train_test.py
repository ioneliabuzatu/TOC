import os
import numpy as np
import config

file_to_split = f"../data/Tran_RGC_scRNA/npys/toy-data/control_and_2w_after_crush_5000x5000_toy.npy"
data = np.load(file_to_split, allow_pickle=True)

if isinstance(data[0, 0], str):
    data = data[1:, :]

len_data = len(data)
indices = np.random.permutation(data.shape[0])
cutoff = int(0.70 * len_data)
training_idx, test_idx = indices[:cutoff], indices[cutoff:]
training_split_data, test_split_data = data[training_idx, :], data[test_idx, :]

print(f"Data has been split in {cutoff} trainset rows and {len_data - cutoff} testset rows.")

path_to_save_split_data = f"../{config.root_stacked_data}/control-2w-split"
if not os.path.exists(path_to_save_split_data):
    os.makedirs(path_to_save_split_data)

np.save(f"../data/Tran_RGC_scRNA/npys/toy-data/split-control_and_2w_after_crush_5000x5000_toy/train.npy", 
        training_split_data)
np.save(f"../data/Tran_RGC_scRNA/npys/toy-data/split-control_and_2w_after_crush_5000x5000_toy/test.npy",
        test_split_data)
