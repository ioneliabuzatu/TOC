import numpy as np
from SERGIO.SERGIO.sergio import sergio
import pandas as pd


def steady_state():
    global sim
    sim = sergio(number_genes=100, number_bins=9, number_sc=30, noise_params=1, decays=0.8, sampling_state=15,
                 noise_type='dpd')
    sim.build_graph(input_file_taregts='SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Interaction_cID_4.txt',
                    input_file_regs='SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Regs_cID_4.txt',
                    shared_coop_state=2)
    sim.simulate()
    expr = sim.getExpressions()
    expr_clean = np.concatenate(expr, axis=1)
    expr_add_outlier_genes = sim.outlier_effect(expr, outlier_prob=0.01, mean=0.8, scale=1)
    """ Add Library Size Effect """
    libFactor, expr_O_L = sim.lib_size_effect(expr_add_outlier_genes, mean=4.6, scale=0.4)
    """ Add Dropouts """
    binary_ind = sim.dropout_indicator(expr_O_L, shape=6.5, percentile=82)
    expr_O_L_D = np.multiply(binary_ind, expr_O_L)
    count_matrix_umi_count_format = sim.convert_to_UMIcounts(expr_O_L_D)
    count_expression_matrix = np.concatenate(count_matrix_umi_count_format, axis=1)
    transpose_count_matrix = count_expression_matrix.T
    print(transpose_count_matrix.shape)


def differentiated_states():
    global sim
    df = pd.read_csv('SERGIO/data_sets/De-noised_100G_6T_300cPerT_dynamics_7_DS6/bMat_cID7.tab', sep='\t', header=None,
                     index_col=None)
    bMat = df.values
    sim = sergio(number_genes=100, number_bins=6, number_sc=10, noise_params=0.2, decays=0.8, sampling_state=1,
                 noise_params_splice=0.07, noise_type='dpd', dynamics=True, bifurcation_matrix=bMat)
    sim.build_graph(
        input_file_taregts='SERGIO/data_sets/De-noised_100G_6T_300cPerT_dynamics_7_DS6/Interaction_cID_7.txt',
        input_file_regs='SERGIO/data_sets/De-noised_100G_6T_300cPerT_dynamics_7_DS6/Regs_cID_7.txt',
        shared_coop_state=2)
    sim.simulate_dynamics()
    exprU, exprS = sim.getExpressions_dynamics()
    exprU_clean = np.concatenate(exprU, axis=1)
    exprS_clean = np.concatenate(exprS, axis=1)
    print(exprU.shape, exprU_clean.shape, exprS_clean.shape)


# steady_state()
differentiated_states()
