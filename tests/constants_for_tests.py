class SteadyStateParams:
    input_file_targets = "./data/2_cells_type_from_De-noised_100G_9T_300cPerT_4_DS1_Interaction_cID_4.txt"
    input_file_regs = "./data/2_cells_type_from_De-noised_100G_9T_300cPerT_4_DS1_Regs_cID_4.txt"
    number_genes = 100
    number_bins = 2
    number_sc = 1
    noise_params = 1
    decays = 0.8
    sampling_state = 3
    noise_type = 'dpd'
    shared_coop_state = 2


class DynamicsStateParams:
    file_targets_dynamics_toy = "./data/2_cells_types_De-noised_100G_6T_300cPerT_dynamics_7_DS6_Interaction_cID_7.txt"
    file_regs_dynamics_toy = "./data/2_cells_types_De-noised_100G_6T_300cPerT_dynamics_7_DS6_Regs_cID_7.txt"
    bmat_file_toy = "./data/2_cells_types_De-noised_100G_6T_300cPerT_dynamics_7_DS6_bMat_cID7.tab"
    # file_targets_dynamics_toy = "../SERGIO/data_sets/De-noised_100G_6T_300cPerT_dynamics_7_DS6/Interaction_cID_7.txt"
    # file_regs_dynamics_toy ="../SERGIO/data_sets/De-noised_100G_6T_300cPerT_dynamics_7_DS6/Regs_cID_7.txt"
    # bmat_file_toy = "../SERGIO/data_sets/De-noised_100G_6T_300cPerT_dynamics_7_DS6/bMat_cID7.tab"
    num_cell_types = 2
    number_sc = 1
    shared_coop_state = 2
    number_genes = 100
    noise_params = 0.2
    decays = 0.8
    sampling_state = 1
    noise_params_splice = 0.07
    noise_type = 'dpd'

