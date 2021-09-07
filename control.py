import time

import jax
import jax.numpy as np

import duckie.sergio_rewrite


def main_control_steady_state():
    start = time.time()
    env = duckie.sergio_rewrite.sergio(
        number_genes=100,  # Features
        number_bins=2,  # Number of cell types
        number_sc=2,  # 0,  # 300,  # Number of single cells for which expression is simulated
        noise_params=1,
        decays=0.8,
        sampling_state=3,  # 15,
    )
    env.build_graph(
        input_file_taregts="duckie/2_cells_type_from_De-noised_100G_9T_300cPerT_4_DS1_Interaction_cID_4.txt",
        input_file_regs="duckie/2_cells_type_from_De-noised_100G_9T_300cPerT_4_DS1_Regs_cID_4.txt",
        shared_coop_state=2
    )

    def loss_fn(actions):
        expr = env.simulate(actions)
        print(f"mean {float(expr.mean().primal):.4f}")
        print(f"std {float(expr.std().primal):.4f}")
        return -np.mean(np.sum(np.power(expr, 2), axis=1))

    actions = np.zeros((env.sampling_state_ * env.num_sc, env.num_bins, env.num_genes))
    # utils.plot(loss_fn, actions)

    loss, grad = jax.value_and_grad(loss_fn)(actions)
    print("loss", loss)
    print(f"grad shape: {grad.shape} \n grad: {grad}")
    actions += 0.1 * -grad
    print(f"Took {time.time() - start:.3f} sec.")


if __name__ == '__main__':
    # main_control_steady_state()
    with jax.disable_jit():
        main_control_steady_state()
