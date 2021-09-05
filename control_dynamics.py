import time

import jax
import jax.numpy as jnp
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import torch
import wandb

import config
from duckie.src.envs import EnvControlDynamics
from models.expert.classfier_cell_state import CellStateClassifier
from models.expert.classfier_cell_state import torch_to_jax


def jax_bce_w_logits(x, y, weight=None, average=True):
    """
    Binary Cross Entropy Loss
    Should be numerically stable, built based on: https://github.com/pytorch/pytorch/issues/751
    :param x: Input tensor
    :param y: Target tensor
    :param weight: Vector of example weights
    :param average: Boolean to average resulting loss vector
    :return: Scalar value
    """
    max_val = jnp.clip(x, 0, None)
    loss = x - x * y + max_val + jnp.log(jnp.exp(-max_val) + jnp.exp((-x - max_val)))

    if weight is not None:
        loss = loss * weight

    if average:
        loss_mean = loss.mean()
        assert loss_mean > 0
        return loss_mean
    else:
        loss_sum = loss.sum()
        assert loss_sum > 0
        return loss_sum


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
    wandb.init(project="TranscriptomicsOptimalControl")

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

    network = CellStateClassifier(num_genes=config.genes_per_single_cell).to("cpu")
    loaded_checkpoint = torch.load(config.checkpoint_filepath, map_location=lambda storage, loc: storage)
    network.load_state_dict(loaded_checkpoint)
    network.eval()
    network = torch_to_jax(network)

    def loss_fn(actions, model=network):
        expression_unspliced, expression_spliced = env_dynamics.step(
            actions,
            ignore_technical_noise=False
        )
        expert_prediction_healthy_state = model(expression_spliced.T[:, :, 1])
        expert_prediction_unhealthy_state = model(expression_spliced.T[:, :, 0])
        truth_diseased = jnp.array(torch.tensor([0.]).unsqueeze(1))
        truth_control = jnp.array(torch.tensor([1.]).unsqueeze(1))
        error_diseased_state = jax_bce_w_logits(expert_prediction_unhealthy_state, truth_diseased)
        error_healthy_state = jax_bce_w_logits(expert_prediction_healthy_state, truth_control)
        return - error_healthy_state - error_diseased_state

    shape = (env_dynamics.env.nBins_, env_dynamics.env.nGenes_)
    actions = jnp.zeros(shape=shape) + 0.1

    for _ in range(1000000):
        loss, grad = jax.value_and_grad(loss_fn)(actions)
        wandb.log({'loss': float(loss)})
        print("loss", loss)
        print(f"grad shape: {grad.shape} \n grad: {grad}")
        actions += 0.001 * -grad
        wandb.log({"gradients cell condition 0": wandb.Histogram(np_histogram=np.histogram(grad[0]))})
        wandb.log({"gradients cell condition 1": wandb.Histogram(np_histogram=np.histogram(grad[1]))})
        wandb.log({"grads_x_genes_x_conditions": wandb.Image(
            sns.heatmap(grad,
                        linewidth=0.5,
                        xticklabels=config.disease_gene_names
                        ))})
        plt.close()

    print(f"Took {time.time() - start:.3f} secs.")


if __name__ == '__main__':
    with jax.disable_jit():
        main_control_dynamics(
            number_genes=12,
            number_cell_types=2,
            number_simulated_cells=1,
            noise_params=1,
            decays=0.8,
            sampling_state=3,
            noise_type='dpd',
            input_file_targets=config.filepath_small_dynamics_targets_diseased,
            input_file_regs=config.filepath_small_dynamics_regulons_diseased,
            bMat=config.filepath_small_dynamics_bifurcation_matrix_diseased,
        )
