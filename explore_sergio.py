import time
import numpy as np
from SERGIO.SERGIO.sergio import sergio
import pandas as pd
import config


def steady_state(number_genes=None,
                 number_bins=None,
                 number_sc=None,
                 noise_params=None,
                 decays=None,
                 sampling_state=None,
                 noise_type=None,
                 input_file_targets=None,
                 input_file_regs=None):
    sim = sergio(number_genes=number_genes,
                 number_bins=number_bins,
                 number_sc=number_sc,
                 noise_params=noise_params,
                 decays=decays,
                 sampling_state=sampling_state,
                 noise_type=noise_type)
    sim.build_graph(input_file_taregts=input_file_targets, input_file_regs=input_file_regs, shared_coop_state=2)
    sim.simulate()
    expression = sim.getExpressions()
    # expr_add_outlier_genes = sim.outlier_effect(expr, outlier_prob=0.01, mean=0.8, scale=1)
    # libFactor, expr_O_L = sim.lib_size_effect(expr_add_outlier_genes, mean=4.6, scale=0.4)
    # binary_ind = sim.dropout_indicator(expr_O_L, shape=6.5, percentile=82)
    # expr_O_L_D = np.multiply(binary_ind, expr_O_L)
    # count_matrix_umi_count_format = sim.convert_to_UMIcounts(expr_O_L_D)
    # count_expression_matrix = np.concatenate(count_matrix_umi_count_format, axis=1)
    # transposed_count_matrix = count_expression_matrix.T
    # return transposed_count_matrix
    return expression


def differentiated_states(bmat_filepath, targets_filepath, regs_filepath, number_of_cell_types):
    df = pd.read_csv(bmat_filepath, sep='\t', header=None, index_col=None)
    bMat = df.values
    sim = sergio(number_genes=100, number_bins=number_of_cell_types, number_sc=2, noise_params=0.2,
                 decays=0.8, sampling_state=1, noise_params_splice=0.07, noise_type='dpd',
                 dynamics=True, bifurcation_matrix=bMat)
    sim.build_graph(input_file_taregts=targets_filepath, input_file_regs=regs_filepath, shared_coop_state=2)
    sim.simulate_dynamics()
    exprU, exprS = sim.getExpressions_dynamics()
    exprU_O, exprS_O = sim.outlier_effect_dynamics(exprU, exprS, outlier_prob=0.01, mean=0.8, scale=1)
    libFactor, exprU_O_L, exprS_O_L = sim.lib_size_effect_dynamics(exprU_O, exprS_O, mean=4.6, scale=0.4)
    binary_indU, binary_indS = sim.dropout_indicator_dynamics(exprU_O_L, exprS_O_L, shape=6.5, percentile=82)
    exprU_O_L_D = np.multiply(binary_indU, exprU_O_L)
    exprS_O_L_D = np.multiply(binary_indS, exprS_O_L)
    count_matrix_U, count_matrix_S = sim.convert_to_UMIcounts_dynamics(exprU_O_L_D, exprS_O_L_D)
    count_matrix_U = np.concatenate(count_matrix_U, axis=1)
    count_matrix_S = np.concatenate(count_matrix_S, axis=1)
    return count_matrix_U, count_matrix_S


if __name__ == "__main__":
    start = time.time()
    steady_state(number_genes=100,
                 number_bins=2,
                 number_sc=10,
                 noise_params=1,
                 decays=0.8,
                 sampling_state=15,
                 noise_type='dpd',
                 input_file_targets="duckie/2_cells_type_from_De-noised_100G_9T_300cPerT_4_DS1_Interaction_cID_4.txt",
                 input_file_regs="duckie/2_cells_type_from_De-noised_100G_9T_300cPerT_4_DS1_Regs_cID_4.txt",
                 )

    # differentiated_states(config.bmat_file_toy,
    #                       config.input_file_targets_dynamics_toy,
    #                       config.input_file_regs_dynamics_toy,
    #                       number_of_cell_types=2)

    print(f"Took {time.time() - start:.4f} sec.")
