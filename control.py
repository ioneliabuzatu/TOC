import time

import jax
import jax.numpy as np

import duckie.duckie


def main():
    # Simulate Clean Data _ Steady-State Simulation
    sim = duckie.duckie.Ducky(
        number_genes=100,  # Features
        number_gene_types=9,  # Number of cell types
        cells_to_simulate=2,  # 0,  # 300,  # Number of single cells for which expression is simulated
        noise_params=1,
        decays=0.8,
        sampling_state=15,
        noise_type='dpd',
        input_file_taregts='duckie/steady-state_input_GRN.txt',
        input_file_regs='duckie/steady-state_input_MRs.txt',
        shared_coop_state=2
    )

    start = time.time()
    sim.simulate()
    expr = sim.getExpressions()

    def loss_fn(x):
        return -np.mean(np.sum(np.power(x, 2), axis=1))

    loss = loss_fn(expr)
    grad = jax.grad(loss_fn)
    print(loss)
    print(grad)
    print(time.time() - start)


if __name__ == '__main__':
    main()
