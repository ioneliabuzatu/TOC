import numpy as np
import pandas as pd
from SERGIO.SERGIO.sergio import sergio

sim = sergio(number_genes=100, number_bins=1, number_sc=300, noise_params=1, decays=0.8, sampling_state=15,
             noise_type='dpd')
sim.build_graph(input_file_taregts='SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Interaction_cID_4.txt',
                input_file_regs='SERGIO/data_sets/De-noised_100G_9T_300cPerT_4_DS1/Regs_cID_4_1cellTypecd.txt',
                shared_coop_state=2)
sim.simulate()
expr = sim.getExpressions()
expr_clean = np.concatenate(expr, axis=1)
