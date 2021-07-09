import os

import experiment_buddy

root_data = "/publicwork/students/buzatu"
root_npys = "data/Tran_RGC_scRNA/npys"
root_toy_data = "data/Tran_RGC_scRNA/npys/toy-data"
root_stacked_data = "data/Tran_RGC_scRNA/npys/stacked"
all_rna_data_npy_filepath = "data/Tran_RGC_scRNA/npys/stacked/control_4d_1w_2w.npy"
PATH_TO_UNZIPPED_CSV_FILES = "data/Tran_RGC_scRNA/unzipped"
root_single_processed_npy_files = "data/Tran_RGC_scRNA/npys/processed-data"
filepath_toy_for_training = root_toy_data + "/control_and_2w_after_crush_5000x5000_toy.npy"
filepath_tmp = root_data + "/control_and_2w_after_crush_9999x24999_toy.npy"
filepath_full_control_2w = os.path.join(root_data, "control_2w.npy")

genes_per_single_cell =24999# 40791 #  5000
epochs = 200
batch_size = 64
lr = 1e-3

experiment_buddy.register(locals())
tensorboard = experiment_buddy.deploy(host="", sweep_yaml="")
