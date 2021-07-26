import numpy as np
from SERGIO.SERGIO.sergio import sergio

sim = sergio(number_genes=23457, number_bins=1, number_sc=10, noise_params=1, decays=0.8, sampling_state=15,
             noise_type='dpd')
sim.build_graph(input_file_taregts='data/scenic/mouse/interaction_cID.txt',
                input_file_regs='data/scenic/mouse/Regs_cID_1cell_dummy.txt',
                shared_coop_state=2)
sim.simulate()
expr = sim.get_last_state()
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
