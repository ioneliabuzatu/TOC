import os

import experiment_buddy

root_data = "/publicwork/students/buzatu"
USE_BUDDY = False
RUN_LOCALLY = False

# dynamics inputs from original code SERGIO
input_file_targets_dynamics = 'SERGIO/data_sets/De-noised_100G_6T_300cPerT_dynamics_7_DS6/Interaction_cID_7.txt'
input_file_regs_dynamics = 'SERGIO/data_sets/De-noised_100G_6T_300cPerT_dynamics_7_DS6/Regs_cID_7.txt'
bmat_file = 'SERGIO/data_sets/De-noised_100G_6T_300cPerT_dynamics_7_DS6/bMat_cID7.tab'
# dynamics toy
input_file_targets_dynamics_toy = "duckie/2_cells_types_De-noised_100G_6T_300cPerT_dynamics_7_DS6_Interaction_cID_7.txt"
input_file_regs_dynamics_toy = "duckie/2_cells_types_De-noised_100G_6T_300cPerT_dynamics_7_DS6_Regs_cID_7.txt"
bmat_file_toy = "duckie/2_cells_types_De-noised_100G_6T_300cPerT_dynamics_7_DS6_bMat_cID7.tab"

# small dynamics data
filepath_small_dynamics_targets="scenicsergio/data/interactions.txt"
filepath_small_dynamics_regulons="scenicsergio/data/regulons.txt"
filepath_small_dynamics_bifurcation_matrix="scenicsergio/data/bifurcation_matrix.tab"

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
filepath_tmp = root_data + "/control_and_2w_after_crush_9999x24999_toy.npy"
filepath_full_control_2w = os.path.join(root_data, "control_2w.npy")

filepath_full_control_2w = os.path.join(root_stacked_data, "control_2w.npy")
filepath_trainset_control_2w = os.path.join(root_stacked_data, "control-2w-split", "trainset_control_2w.npy")
filepath_testset_control_2w = os.path.join(root_stacked_data, "control-2w-split", "testset_control_2w.npy")
filepath_train_toy = "data/Tran_RGC_scRNA/npys/toy-data/split-control_and_2w_after_crush_5000x5000_toy/train.npy"
filepath_test_toy = "data/Tran_RGC_scRNA/npys/toy-data/split-control_and_2w_after_crush_5000x5000_toy/test.npy"

filepath_12_genes = "./control_disease_12_genes_expressions.npy"
gene_names = ["Ank2", "Cartpt", "Chrm2", "Eomes", "Kctd4", "Mafb", "Neurod2", "Opn4","Pou4f3", "Ttn", "mt-Cytb","mt-Nd4"]

checkpoint_filepath_classifier = os.path.join("models/checkpoints")
checkpoint_filepath = os.path.join("models/checkpoints", "classifier_12_genes.pth")

genes_per_single_cell = 12  # 40790 #  5000
epochs = 300
batch_size = 64
lr = 1e-3

if USE_BUDDY:
    experiment_buddy.register_defaults(locals())
    tensorboard = experiment_buddy.deploy(
        "",
        sweep_yaml="",
        proc_num=1,
    )