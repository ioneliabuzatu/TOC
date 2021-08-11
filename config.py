import os

# dynamics inputs from original code SERGIO
input_file_targets_dynamics = 'SERGIO/data_sets/De-noised_100G_6T_300cPerT_dynamics_7_DS6/Interaction_cID_7.txt'
input_file_regs_dynamics = 'SERGIO/data_sets/De-noised_100G_6T_300cPerT_dynamics_7_DS6/Regs_cID_7.txt'
bmat_file = 'SERGIO/data_sets/De-noised_100G_6T_300cPerT_dynamics_7_DS6/bMat_cID7.tab'
# dynamics toy
input_file_targets_dynamics_toy = "duckie/2_cells_types_De-noised_100G_6T_300cPerT_dynamics_7_DS6_Interaction_cID_7.txt"
input_file_regs_dynamics_toy = "duckie/2_cells_types_De-noised_100G_6T_300cPerT_dynamics_7_DS6_Regs_cID_7.txt"
bmat_file_toy = "duckie/2_cells_types_De-noised_100G_6T_300cPerT_dynamics_7_DS6_bMat_cID7.tab"

# preprocess_workflow.py user inputs
filepath_adjancies_control = "data/scenic/mouse/control/GSE133382.adjacencies.npy"
filepath_adjancies_disease = "data/scenic/mouse/diseased/GSE133382.disease.adjacencies.npy"

root_npys = "data/Tran_RGC_scRNA/npys"
root_toy_data = "data/Tran_RGC_scRNA/npys/toy-data"
root_stacked_data = "data/Tran_RGC_scRNA/npys/stacked"
all_rna_data_npy_filepath = "data/Tran_RGC_scRNA/npys/stacked/control_4d_1w_2w.npy"
PATH_TO_UNZIPPED_CSV_FILES = "data/Tran_RGC_scRNA/unzipped"
root_single_processed_npy_files = "data/Tran_RGC_scRNA/npys/processed-data"
filepath_toy_for_training = root_toy_data + "/control_and_2w_after_crush_5000x5000_toy.npy"
filepath_full_control_2w = os.path.join(root_stacked_data, "control_2w.npy")
filepath_trainset_control_2w = os.path.join(root_stacked_data, "control-2w-split", "trainset_control_2w.npy")
filepath_testset_control_2w = os.path.join(root_stacked_data, "control-2w-split", "testset_control_2w.npy")
filepath_train_toy = "data/Tran_RGC_scRNA/npys/toy-data/split-control_and_2w_after_crush_5000x5000_toy/train.npy"
filepath_test_toy = "data/Tran_RGC_scRNA/npys/toy-data/split-control_and_2w_after_crush_5000x5000_toy/test.npy"

checkpoint_filepath_classifier = os.path.join("models/checkpoints")

genes_per_single_cell = 5000  # 40790 #  5000
epochs = 100
batch_size = 64
lr = 1e-3
