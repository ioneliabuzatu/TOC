import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from classfier_cell_state import CellStateClassifier
from classfier_cell_state import TranscriptomicsDataset


@torch.no_grad()
def binary_acc(logits, targets):
    if logits.dim != targets.dim:
        targets = targets.unsqueeze(1)
    y_pred_tag = torch.round(torch.sigmoid(logits))

    correct_results_sum = (y_pred_tag == targets).sum().float()
    acc = correct_results_sum / targets.shape[0]
    acc = torch.round(acc * 100)
    return acc.item()


@torch.no_grad()
def get_accuracy(network, dataloader):
    network.eval()
    device = next(network.parameters()).device
    accuracies = []
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = network(inputs)
        accuracy_mini_batch = binary_acc(logits, targets)
        accuracies.append(accuracy_mini_batch)

    return accuracies


network = CellStateClassifier(num_genes=config.genes_per_single_cell)
network.load_state_dict(torch.load("models/checkpoints/classifier_5kgenes.pth"))
testset = TranscriptomicsDataset(filepath_data=config.filepath_test_toy, device="cuda")
dataloader = DataLoader(testset, batch_size=1, shuffle=False)
network.eval()
print(np.asarray(get_accuracy(network, dataloader)).mean())
