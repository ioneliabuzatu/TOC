import jax

import duckie.sergio_control
import jax.numpy as jnp
import pandas as pd


class EnvControlSteadyState(object):
    def __init__(self, number_genes, number_bins, number_sc, noise_params, decays, sampling_state, noise_type,
                 input_file_targets, input_file_regs,
                 shared_coop_state):
        self.env = duckie.sergio_control.sergio(
            number_genes=number_genes,
            number_bins=number_bins,
            number_sc=number_sc,
            noise_params=noise_params,
            decays=decays,
            sampling_state=sampling_state,
            noise_type=noise_type,
        )

        self.env.build_graph(input_file_targets, input_file_regs, shared_coop_state)

    def step(self, actions, ignore_technical_noise=False):
        self.simulate(actions)
        expression = self.get_expressions()
        if ignore_technical_noise:
            return expression
        return self.add_technical_noise(expression)

    def simulate(self, actions):
        return self.env.simulate(actions)

    def get_expressions(self):
        return self.env.get_expressions()

    def add_technical_noise(self, expression):
        key = jax.random.PRNGKey(0)

        kay, key = jax.random.split(key, num=2)
        expr_add_outlier_genes, key = self.env.outlier_effect(expression, outlier_prob=0.01, mean=0.8, scale=1, key=kay)

        kay, key = jax.random.split(key, num=2)
        _, expression_with_outliers = self.env.lib_size_effect(expr_add_outlier_genes, mean=4.6, scale=0.4, key=kay)

        kay, key = jax.random.split(key, num=2)
        binary_dropout_indices = self.env.dropout_indicator(expression_with_outliers, shape=6.5, percentile=82)

        kay, key = jax.random.split(key, num=2)
        expression_with_outliers_and_dropout = jnp.multiply(binary_dropout_indices, expression_with_outliers)

        kay, key = jax.random.split(key, num=2)
        count_matrix_umi_count_format = self.env.convert_to_UMIcounts_continuous(expression_with_outliers_and_dropout, kay)

        count_expression_matrix = jnp.concatenate(count_matrix_umi_count_format, axis=1)

        transpose_count_matrix = count_expression_matrix.T
        return transpose_count_matrix


class EnvDifferentiationDynamics:
    def __init__(self, bmat_file, input_file_targets, input_file_regs, shared_coop_state):
        df = pd.read_csv(bmat_file, sep='\t', header=None, index_col=None)
        bMat = df.values
        self.env = duckie.sergio_control.sergio(number_genes=100, number_bins=6, number_sc=10, noise_params=0.2,
                                                decays=0.8,
                                                sampling_state=1,
                                                noise_params_splice=0.07, noise_type='dpd', dynamics=True,
                                                bifurcation_matrix=bMat)
        self.env.build_graph(input_file_targets, input_file_regs, shared_coop_state)

    def step(self, ignore_technical_noise=False) -> tuple:
        self.simulate_dynamics()
        exprU, exprS = self.get_expression_dynamics()
        if ignore_technical_noise:
            return (exprU, exprS)
        count_matrix_unpliced, count_matrix_spliced = self.add_technical_noise(exprU, exprS)
        return count_matrix_unpliced, count_matrix_spliced

    def simulate_dynamics(self):
        self.env.simulate_dynamics()

    def get_expression_dynamics(self):
        exprU, exprS = self.env.getExpressions_dynamics()
        return exprU, exprS

    def add_technical_noise(self, exprU, exprS):
        exprU_O, exprS_O = self.env.outlier_effect_dynamics(exprU, exprS, outlier_prob=0.01, mean=0.8, scale=1)
        libFactor, exprU_O_L, exprS_O_L = self.env.lib_size_effect_dynamics(exprU_O, exprS_O, mean=4.6, scale=0.4)
        binary_indU, binary_indS = self.env.dropout_indicator_dynamics(exprU_O_L, exprS_O_L, shape=6.5, percentile=82)
        exprU_O_L_D = jnp.multiply(binary_indU, exprU_O_L)
        exprS_O_L_D = jnp.multiply(binary_indS, exprS_O_L)
        count_matrix_U, count_matrix_S = self.env.convert_to_UMIcounts_dynamics(exprU_O_L_D, exprS_O_L_D)
        count_matrix_unspliced = jnp.concatenate(count_matrix_U, axis=1)
        count_matrix_spliced = jnp.concatenate(count_matrix_S, axis=1)
        return count_matrix_unspliced, count_matrix_spliced
