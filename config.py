import experiment_buddy
root_npys = "Tran_RGC_scRNA/npys"
all_rna_data_npy_filepath = "Tran_RGC_scRNA/npys/stacked/control_4d_1w_2w.npy"
root_toy_data = "Tran_RGC_scRNA/npys/toy-data"
PATH_TO_UNZIPPED_CSV_FILES = "Tran_RGC_scRNA/unzipped"
root_single_processed_npy_files = "Tran_RGC_scRNA/npys/processed-data"
filepath_toy_for_training = root_toy_data + "/control_and_2w_after_crush_100x1000toy.npy"
genes_per_single_cell = 1000
epochs = 1000

experiment_buddy.register(locals())
tensorboard = experiment_buddy.deploy(host="", sweep_yaml="")