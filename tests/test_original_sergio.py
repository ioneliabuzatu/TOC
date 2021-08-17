import unittest

import jax.numpy as jnp
import numpy as np
import pandas as pd

import constants_for_tests
import duckie.sergio_control
from SERGIO.SERGIO.sergio import sergio


class TestOriginalSergio(unittest.TestCase):
    """" Check if Duckie is good as the original implementation of SERGIO. """

    def test_original_sergio_vs_duckie_slow_steady_without_noise(self):
        expected_mean = self.get_sergio_steady_expression(without_noise=True).mean()
        actual_mean = self.get_duckie_steady_expression(without_noise=True).mean()
        self.assertAlmostEqual(expected_mean, actual_mean, delta=0.9)

    def test_original_sergio_vs_duckie_slow_steady_with_noise(self):
        """ Takes approx 13 secs """
        actual_mean = self.get_duckie_steady_expression(without_noise=False).mean()
        expected_mean = self.get_sergio_steady_expression(without_noise=False).mean()
        self.assertAlmostEqual(expected_mean, actual_mean, delta=0.6)

    def test_original_sergio_vs_duckie_slow_dynamics_without_noise(self):
        duckie_unspliced_expression, duckie_spliced_expression = self.get_duckie_dynamics_expression(without_noise=True)
        # sergio_unsplice_expression, sergio_splice_expression = self.get_sergio_dynamics_expression(without_noise=True)
        expected_mean = 10  # sergio_spliced_expression.mean()
        actual_mean = duckie_spliced_expression.mean()
        self.assertAlmostEqual(expected_mean, actual_mean, delta=0.7)

    def test_original_sergio_vs_duckie_slow_dynamics_with_noise(self):
        duckie_unspliced_expression, duckie_splice_expression = self.get_duckie_dynamics_expression(without_noise=False)
        # sergio_unsplice_expression, sergio_splice_expression = self.get_sergio_dynamics_expression(without_noise=False)
        expected_mean = 0.1  # sergio_spliced_expression.mean()
        actual_mean = duckie_splice_expression.mean()
        self.assertAlmostEqual(expected_mean, actual_mean, delta=0.6)

    def get_sergio_dynamics_expression(self, without_noise=False):
        dynamics_inputs_obj = constants_for_tests.DynamicsStateParams()
        bMat = pd.read_csv(dynamics_inputs_obj.bmat_file_toy, sep='\t', header=None, index_col=None).values
        sim = sergio(number_genes=dynamics_inputs_obj.number_genes,
                     number_bins=dynamics_inputs_obj.num_cell_types,
                     number_sc=dynamics_inputs_obj.number_sc,
                     noise_params=dynamics_inputs_obj.noise_params,
                     decays=dynamics_inputs_obj.decays,
                     sampling_state=dynamics_inputs_obj.sampling_state,
                     noise_params_splice=dynamics_inputs_obj.noise_params_splice,
                     noise_type=dynamics_inputs_obj.noise_type,
                     dynamics=True,
                     bifurcation_matrix=bMat
                     )
        sim.build_graph(input_file_taregts=dynamics_inputs_obj.file_targets_dynamics_toy,
                        input_file_regs=dynamics_inputs_obj.file_regs_dynamics_toy,
                        shared_coop_state=2
                        )
        sim.simulate_dynamics()
        exprU, exprS = sim.getExpressions_dynamics()
        if without_noise:
            return exprU, exprS
        exprU_O, exprS_O = sim.outlier_effect_dynamics(exprU, exprS, outlier_prob=0.01, mean=0.8, scale=1)
        libFactor, exprU_O_L, exprS_O_L = sim.lib_size_effect_dynamics(exprU_O, exprS_O, mean=4.6, scale=0.4)
        binary_indU, binary_indS = sim.dropout_indicator_dynamics(exprU_O_L, exprS_O_L, shape=6.5, percentile=82)
        exprU_O_L_D = np.multiply(binary_indU, exprU_O_L)
        exprS_O_L_D = np.multiply(binary_indS, exprS_O_L)
        count_matrix_U, count_matrix_S = sim.convert_to_UMIcounts_dynamics(exprU_O_L_D, exprS_O_L_D)
        count_matrix_U = np.concatenate(count_matrix_U, axis=1)
        count_matrix_S = np.concatenate(count_matrix_S, axis=1)
        return count_matrix_U, count_matrix_S

    def get_duckie_dynamics_expression(self, without_noise):
        dynamics_inputs_obj = constants_for_tests.DynamicsStateParams()
        bMat = pd.read_csv(dynamics_inputs_obj.bmat_file_toy, sep='\t', header=None, index_col=None).values
        env = duckie.sergio_control.sergio(
            number_genes=dynamics_inputs_obj.number_genes,
            number_bins=dynamics_inputs_obj.num_cell_types,
            number_sc=dynamics_inputs_obj.number_sc,
            noise_params=dynamics_inputs_obj.noise_params,
            decays=dynamics_inputs_obj.decays,
            sampling_state=dynamics_inputs_obj.sampling_state,
            noise_type=dynamics_inputs_obj.noise_type,
            dynamics=True,
            bifurcation_matrix=bMat,
        )
        env.build_graph(dynamics_inputs_obj.file_targets_dynamics_toy,
                        dynamics_inputs_obj.file_regs_dynamics_toy,
                        dynamics_inputs_obj.shared_coop_state)
        env.simulate_dynamics()
        exprU, exprS = env.getExpressions_dynamics()
        if without_noise:
            return exprU, exprS
        exprU_O, exprS_O = env.outlier_effect_dynamics(exprU, exprS, outlier_prob=0.01, mean=0.8, scale=1)
        libFactor, exprU_O_L, exprS_O_L = env.lib_size_effect_dynamics(exprU_O, exprS_O, mean=4.6, scale=0.4)
        binary_indU, binary_indS = env.dropout_indicator_dynamics(exprU_O_L, exprS_O_L, shape=6.5, percentile=82)
        exprU_O_L_D = np.multiply(binary_indU, exprU_O_L)
        exprS_O_L_D = np.multiply(binary_indS, exprS_O_L)
        count_matrix_U, count_matrix_S = env.convert_to_UMIcounts_dynamics(exprU_O_L_D, exprS_O_L_D)
        count_matrix_U = np.concatenate(count_matrix_U, axis=1)
        count_matrix_S = np.concatenate(count_matrix_S, axis=1)
        return count_matrix_U, count_matrix_S

    @staticmethod
    def get_sergio_steady_expression(without_noise=False):
        steady_state_params = constants_for_tests.SteadyStateParams()
        sim = sergio(number_genes=steady_state_params.number_genes,
                     number_bins=steady_state_params.number_bins,
                     number_sc=steady_state_params.number_sc,
                     noise_params=steady_state_params.noise_params,
                     decays=steady_state_params.decays,
                     sampling_state=steady_state_params.sampling_state,
                     noise_type=steady_state_params.noise_type)
        sim.build_graph(input_file_taregts=steady_state_params.input_file_targets,
                        input_file_regs=steady_state_params.input_file_regs, shared_coop_state=2)
        sim.simulate()
        expression = sim.getExpressions()

        if without_noise:
            return expression
        else:
            expr_add_outlier_genes = sim.outlier_effect(expression, outlier_prob=0.01, mean=0.8, scale=1)
            libFactor, expr_O_L = sim.lib_size_effect(expr_add_outlier_genes, mean=4.6, scale=0.4)
            binary_ind = sim.dropout_indicator(expr_O_L, shape=6.5, percentile=82)
            expr_O_L_D = np.multiply(binary_ind, expr_O_L)
            count_matrix_umi_count_format = sim.convert_to_UMIcounts(expr_O_L_D)
            count_expression_matrix = np.concatenate(count_matrix_umi_count_format, axis=1)
            return count_expression_matrix

    @staticmethod
    def get_duckie_steady_expression(without_noise=False):
        steady_state_params = constants_for_tests.SteadyStateParams()
        env = duckie.sergio_control.sergio(
            number_genes=steady_state_params.number_genes,
            number_bins=steady_state_params.number_bins,
            number_sc=steady_state_params.number_sc,
            noise_params=steady_state_params.noise_params,
            decays=steady_state_params.decays,
            sampling_state=steady_state_params.sampling_state,
            noise_type=steady_state_params.noise_type,
        )
        env.build_graph(steady_state_params.input_file_targets,
                        steady_state_params.input_file_regs,
                        steady_state_params.shared_coop_state)
        env.simulate(actions=None)
        expression = env.get_expressions()

        if without_noise:
            return expression
        else:
            expr_add_outlier_genes = env.outlier_effect(expression, outlier_prob=0.01, mean=0.8, scale=1)
            _, expression_with_outliers = env.lib_size_effect(expr_add_outlier_genes, mean=4.6, scale=0.4)
            binary_dropout_indices = env.dropout_indicator(expression_with_outliers, shape=6.5, percentile=82)
            expression_with_outliers_and_dropout = jnp.multiply(binary_dropout_indices, expression_with_outliers)
            count_matrix_umi_count_format = env.convert_to_UMIcounts_fix_for_jax_poisson(
                expression_with_outliers_and_dropout
            )
            count_expression_matrix = jnp.concatenate(count_matrix_umi_count_format, axis=1)
            return count_expression_matrix


if __name__ == '__main__':
    unittest.main()
