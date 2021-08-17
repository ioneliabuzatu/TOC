import unittest

import jax.numpy as jnp
import numpy as np

import constants_for_tests
import duckie.sergio_control
from SERGIO.SERGIO.sergio import sergio


class TestOriginalSergio(unittest.TestCase):
    """" Check if Duckie is good as the original implementation of SERGIO. """

    def test_original_sergio_vs_duckie_slow_steady_no_noise(self):
        expected_mean = self.get_sergio_steady_expression(without_noise=True).mean()
        actual_mean = self.get_duckie_steady_expression(without_noise=True).mean()
        self.assertAlmostEqual(expected_mean, actual_mean, delta=0.9)

    def test_original_sergio_vs_duckie_slow_steady_with_noise(self):
        actual_mean = self.get_duckie_steady_expression(without_noise=False).mean()
        expected_mean = self.get_sergio_steady_expression(without_noise=False).mean()
        self.assertAlmostEqual(expected_mean, actual_mean, delta=0.6)

    @staticmethod
    def get_sergio_steady_expression(without_noise=False):
        sim = sergio(number_genes=constants_for_tests.number_genes,
                     number_bins=constants_for_tests.number_bins,
                     number_sc=constants_for_tests.number_sc,
                     noise_params=constants_for_tests.noise_params,
                     decays=constants_for_tests.decays,
                     sampling_state=constants_for_tests.sampling_state,
                     noise_type=constants_for_tests.noise_type)
        sim.build_graph(input_file_taregts=constants_for_tests.input_file_targets,
                        input_file_regs=constants_for_tests.input_file_regs, shared_coop_state=2)
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
            transposed_count_matrix = count_expression_matrix.T
            return transposed_count_matrix

    @staticmethod
    def get_duckie_steady_expression(without_noise=False):
        env = duckie.sergio_control.sergio(
            number_genes=constants_for_tests.number_genes,
            number_bins=constants_for_tests.number_bins,
            number_sc=constants_for_tests.number_sc,
            noise_params=constants_for_tests.noise_params,
            decays=constants_for_tests.decays,
            sampling_state=constants_for_tests.sampling_state,
            noise_type=constants_for_tests.noise_type,
        )
        env.build_graph(constants_for_tests.input_file_targets,
                        constants_for_tests.input_file_regs,
                        constants_for_tests.shared_coop_state)
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
            transpose_count_matrix = count_expression_matrix.T
            return transpose_count_matrix


if __name__ == '__main__':
    unittest.main()
