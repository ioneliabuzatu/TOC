import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import config
from classfier_cell_state import update
from classfier_cell_state import evaluate
from classfier_cell_state import CellStateClassifier
from classfier_cell_state import TranscriptomicsDataset
from classfier_cell_state import train_val_dataset
from classfier_cell_state import get_accuracy


def train(epochs=200):
    torch.manual_seed(1806)
    torch.cuda.manual_seed(1806)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Run on {device}")

    writer = config.tensorboard

    dataset = TranscriptomicsDataset(filepath_data=config.filepath_toy_for_training)
    train_and_val_dataset = train_val_dataset(dataset, val_split=0.25)
    train_dataloader = DataLoader(train_and_val_dataset["train"], batch_size=64, shuffle=True)
    val_dataloader = DataLoader(train_and_val_dataset["val"], batch_size=64, shuffle=True)

    network = CellStateClassifier(num_genes=config.genes_per_single_cell).to(device)
    sgd = optim.SGD(network.parameters(), lr=1e-3, momentum=0.9)
    criterium = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(epochs)):
        train_errors = update(network, train_dataloader, criterium, sgd)
        writer.add_scalar("TrainingLoss", np.asarray(train_errors).mean(), epoch)
        val_errors = evaluate(network, val_dataloader, criterium)
        writer.add_scalar("ValidationLoss", np.asarray(val_errors).mean(), epoch)
        accuracies = get_accuracy(network, val_dataloader)
        writer.add_scalar("Accuracy", np.asarray(accuracies).mean(), epoch)


if __name__ == "__main__":
    train(epochs=config.epochs)
