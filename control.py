import time

import jax
import jax.numpy as np

import duckie.sergio_control



def main_control_steady_state():
    start = time.time()
    env = duckie.sergio_control.sergio(
        number_genes=100,  # Features
        number_bins=2,  # Number of cell types
        number_sc=2,  # 0,  # 300,  # Number of single cells for which expression is simulated
        noise_params=1,
        decays=0.8,
        sampling_state=3,  # 15,
        noise_type='dpd',
    )
    env.build_graph(
        input_file_taregts="duckie/2_cells_type_from_De-noised_100G_9T_300cPerT_4_DS1_Interaction_cID_4.txt",
        input_file_regs="duckie/2_cells_type_from_De-noised_100G_9T_300cPerT_4_DS1_Regs_cID_4.txt",
        shared_coop_state=2
    )

    def loss_fn(actions):
        env.simulate(actions)
        expression = env.getExpressions()

        expr_add_outlier_genes = env.outlier_effect(expression, outlier_prob=0.01, mean=0.8, scale=1)
        _, expression_with_outliers = env.lib_size_effect(expr_add_outlier_genes, mean=4.6, scale=0.4)
        binary_dropout_indices = env.dropout_indicator(expression_with_outliers, shape=6.5, percentile=82)
        expression_with_outliers_and_dropout = np.multiply(binary_dropout_indices, expression_with_outliers)
        count_matrix_umi_count_format = env.convert_to_UMIcounts(expression_with_outliers_and_dropout)
        count_expression_matrix = np.concatenate(count_matrix_umi_count_format, axis=1)
        transpose_count_matrix = count_expression_matrix.T

        return -np.mean(np.sum(np.power(expression, 2), axis=1))

    actions = np.zeros((env.sampling_state_ * env.nSC_, env.nBins_, env.nGenes_))
    loss, grad = jax.value_and_grad(loss_fn)(actions)
    print("loss", loss)
    print(f"grad shape: {grad.shape} \n grad: {grad}")
    actions += 0.1 * -grad
    print(f"Took {time.time() - start:.3f} sec.")


if __name__ == '__main__':
    with jax.disable_jit():
        main_control_steady_state()
