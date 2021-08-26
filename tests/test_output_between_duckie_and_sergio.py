from unittest import TestCase

import numpy as np

import duckie.sergio_rewrite
from SERGIO.SERGIO.sergio import sergio


class TestOutputComparison(TestCase):
    def test_steady_state(self):
        init_shared_params = {"number_genes":100, "number_bins":2, "number_sc":1, "noise_params":1, "decays":0.8}
        build_graph_shared_params = {
        "input_file_taregts": "../duckie/2_cells_type_from_De-noised_100G_9T_300cPerT_4_DS1_Interaction_cID_4.txt",
        "input_file_regs": "../duckie/2_cells_type_from_De-noised_100G_9T_300cPerT_4_DS1_Regs_cID_4.txt",
        "shared_coop_state": 2
        }

        # duckie env
        env_duckie = duckie.sergio_rewrite.sergio(sampling_state=3, **init_shared_params)
        env_duckie.build_graph(**build_graph_shared_params)
        actions = np.zeros((env_duckie.sampling_state_ * env_duckie.nSC_, env_duckie.nBins_, env_duckie.nGenes_))
        expression_duckie = env_duckie.simulate(actions)
        mean_expression_duckie = expression_duckie.mean()
        print(f"duckie mean expression: {mean_expression_duckie}")

        # sergio env
        sim = sergio(sampling_state=15, noise_type='dpd', **init_shared_params)
        sim.build_graph(**build_graph_shared_params)
        sim.simulate()
        expression_sergio = sim.getExpressions()
        mean_expression_sergio = expression_sergio.mean()
        print(f"sergio mean expression: {mean_expression_sergio}")

        self.assertAlmostEqual(mean_expression_sergio, float(mean_expression_duckie), delta=0.9)
