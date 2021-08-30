import sys
import time

import jax
import jax.numpy as jnp

import config
from duckie.src.envs import EnvControlDynamics


def main_control_dynamics(number_genes,
                          number_cell_types,
                          number_simulated_cells,
                          noise_params,
                          decays,
                          sampling_state,
                          noise_type,
                          input_file_targets,
                          input_file_regs,
                          bMat: str,
                          ):
    start = time.time()
    env_dynamics = EnvControlDynamics(
        number_genes=number_genes,
        number_bins=number_cell_types,
        number_sc=number_simulated_cells,
        noise_params=noise_params,
        decays=decays,
        sampling_state=sampling_state,
        noise_type=noise_type,
        input_file_targets=input_file_targets,
        input_file_regs=input_file_regs,
        bmat_file=bMat,
        shared_coop_state=2,
        )

    def loss_fn(actions):
        expression_unspliced, expression_spliced = env_dynamics.step(
            actions,
            ignore_technical_noise=False
            )
        return -jnp.mean(jnp.sum(jnp.power(expression_spliced, 2), axis=1))

    shape = (env_dynamics.env.sampling_state_ * env_dynamics.env.nSC_, env_dynamics.env.nBins_, env_dynamics.env.nGenes_)
    actions = jax.random.normal(shape=shape, key=env_dynamics.env.create_kay) + 0.1

    for _ in range(10):
        loss, grad = jax.value_and_grad(loss_fn)(actions)
        print("loss", loss)
        print(f"grad shape: {grad.shape} \n grad: {grad}")
        actions += 0.1 * -grad

    print(f"Took {time.time() - start:.3f} secs.")


if __name__ == '__main__':
    with jax.disable_jit():
        main_control_dynamics(number_genes=12,
                              number_cell_types=2,
                              number_simulated_cells=1,
                              noise_params=1,
                              decays=0.8,
                              sampling_state=3,
                              noise_type='dpd',
                              input_file_targets=config.filepath_small_dynamics_targets,
                              input_file_regs=config.filepath_small_dynamics_regulons,
                              bMat=config.filepath_small_dynamics_bifurcation_matrix,
                              )
