import time

import jax
import jax.numpy as jnp

from duckie.src.envs import EnvControlSteadyState


def main_control_steady_state():
    start = time.time()
    env_steady_state = EnvControlSteadyState(
        number_genes=100,  # Features
        number_bins=2,  # Number of cell types
        number_sc=2,  # 0,  # 300,  # Number of single cells for which expression is simulated
        noise_params=1,
        decays=0.8,
        sampling_state=3,  # 15,
        noise_type='dpd',
        input_file_targets="duckie/2_cells_type_from_De-noised_100G_9T_300cPerT_4_DS1_Interaction_cID_4.txt",
        input_file_regs="duckie/2_cells_type_from_De-noised_100G_9T_300cPerT_4_DS1_Regs_cID_4.txt",
        shared_coop_state=2
    )

    def loss_fn(actions):
        count_matrix_expression = env_steady_state.step(actions, ignore_technical_noise=False)
        return -jnp.mean(jnp.sum(jnp.power(count_matrix_expression, 2), axis=1))

    actions = jnp.zeros((env_steady_state.env.sampling_state_ * env_steady_state.env.nSC_, env_steady_state.env.nBins_,
                         env_steady_state.env.nGenes_))
    loss, grad = jax.value_and_grad(loss_fn)(actions)
    print("loss", loss)
    print(f"grad shape: {grad.shape} \n grad: {grad}")
    actions += 0.1 * -grad
    print(f"Took {time.time() - start:.3f} sec.")


if __name__ == '__main__':
    with jax.disable_jit():
        main_control_steady_state()
