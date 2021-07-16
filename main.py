import config
import torch.nn as nn
from classfier_cell_state import CellStateClassifier
import numpy as np
import torch
from function_g import Controller
import matplotlib.pyplot as plt


def main(device, time_steps=10):
    losses = []

    network = CellStateClassifier(num_genes=4).to(device)

    function_g = Controller(4)
    optimizer_name = "Adam"
    optimizer_parameters = {"lr": 0.000625, "eps": 0.00015}
    optimizer = getattr(torch.optim, optimizer_name)(function_g.parameters(), **optimizer_parameters)

    for time_step in range(time_steps):
        toy_input = create_toy_input()

        # a function G that changes the sequence
        changed_seq_t = function_g(toy_input).to(device)

        cost = cost_function(network, changed_seq_t, true_label=torch.ones(10))
        losses.append(cost.detach().numpy() if device.type == "cpu" else cost.item())

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

    plt.plot(losses)
    plt.show()


def create_toy_input(genes: int = 4) -> torch.Tensor:
    toy_input = np.zeros((10, genes))
    genes_expressed = np.random.randint(0, genes, genes // 2)
    toy_input[:, genes_expressed] = 1
    return torch.from_numpy(toy_input).float()


def cost_function(classifier, cell_rna_content, true_label):
    return nn.BCEWithLogitsLoss()(classifier(cell_rna_content).squeeze(), true_label)


def random_control_discrete(sequence):
    mask = torch.FloatTensor(sequence.shape).uniform_() > 0.8
    return sequence * mask


def random_control_continuous(sequence: torch.Tensor) -> torch.Tensor:
    mask = torch.randint(0, 300, sequence.shape)
    return sequence + mask


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: " + str(device))
    main(device)
