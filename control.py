import time

import jax
import jax.numpy as np

import duckie.duckie


def main():
    # Simulate Clean Data _ Steady-State Simulation
    start = time.time()
    env = duckie.duckie.Ducky(
        number_genes=100,  # Features
        number_gene_types=9,  # Number of cell types
        cells_to_simulate=2,  # 0,  # 300,  # Number of single cells for which expression is simulated
        noise_params=1,
        decays=0.8,
        sampling_state=3,  # 15,
        noise_type='dpd',
        input_file_taregts='duckie/steady-state_input_GRN.txt',
        input_file_regs='duckie/steady-state_input_MRs.txt',
        shared_coop_state=2
    )

    def loss_fn(actions):
        env = duckie.duckie.Ducky(
            number_genes=100,  # Features
            number_gene_types=9,  # Number of cell types
            cells_to_simulate=2,  # 0,  # 300,  # Number of single cells for which expression is simulated
            noise_params=1,
            decays=0.8,
            sampling_state=3,  # 15,
            noise_type='dpd',
            input_file_taregts='duckie/steady-state_input_GRN.txt',
            input_file_regs='duckie/steady-state_input_MRs.txt',
            shared_coop_state=2
        )
        env.simulate(actions)
        expr = env.get_last_state()
        return -np.mean(np.sum(np.power(expr, 2), axis=1))

    actions = np.zeros((env.simulation_length, env.num_cell_types, env.num_genes))
    loss = loss_fn(actions)
    grad_fn = jax.grad(loss_fn)
    print("loss", loss)
    print(grad_fn(actions))
    actions = actions + 0.1 * -grad_fn(actions)
    print(time.time() - start)


if __name__ == '__main__':
    with jax.disable_jit():
        main()
