import numpy as np
from SERGIO.SERGIO.sergio import sergio

sim = sergio(number_genes=100, number_bins=1, number_sc=10, noise_params=1, decays=0.8, sampling_state=15,
             noise_type='dpd')
sim.build_graph(input_file_taregts='SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Interaction_cID_4.txt',
                input_file_regs='Regs_cID_4_1cellType.txt',
                shared_coop_state=2)
sim.simulate()
expr = sim.getExpressions()
expr_clean = np.concatenate(expr, axis=1)

"""
Add outlier genes
"""
expr_O = sim.outlier_effect(expr, outlier_prob=0.01, mean=0.8, scale=1)

"""
Add Library Size Effect
"""
libFactor, expr_O_L = sim.lib_size_effect(expr_O, mean=4.6, scale=0.4)

"""
Add Dropouts
"""
binary_ind = sim.dropout_indicator(expr_O_L, shape=6.5, percentile=82)
expr_O_L_D = np.multiply(binary_ind, expr_O_L)

"""
Convert to UMI count
"""
count_matrix = sim.convert_to_UMIcounts(expr_O_L_D)

"""
Make a 2d gene expression matrix
"""
count_matrix = np.concatenate(count_matrix, axis=1)

transpose_count_matrix = count_matrix.T

print()
