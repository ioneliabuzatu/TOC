import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from classfier_cell_state import CellStateClassifier
from classfier_cell_state import TranscriptomicsDataset
from classfier_cell_state import evaluate
from classfier_cell_state import get_accuracy
from classfier_cell_state import train_val_dataset
from classfier_cell_state import update


def train(filepath_training_data, epochs=200):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Run on {device}")

    torch.manual_seed(1806)
    if device == "cuda":
        torch.cuda.manual_seed(1806)

    writer = config.tensorboard

    dataset = TranscriptomicsDataset(filepath_data=filepath_training_data)
    train_and_val_dataset = train_val_dataset(dataset, val_split=0.25)
    train_dataloader = DataLoader(train_and_val_dataset["train"], batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(train_and_val_dataset["val"], batch_size=config.batch_size, shuffle=True)

    network = CellStateClassifier(num_genes=config.genes_per_single_cell).to(device)
    sgd = optim.SGD(network.parameters(), lr=config.lr, momentum=0.9)
    criterium = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(epochs)):
        train_errors = update(network, train_dataloader, criterium, sgd)
        writer.add_scalar("TrainingLoss", np.asarray(train_errors).mean(), epoch)
        val_errors = evaluate(network, val_dataloader, criterium)
        writer.add_scalar("ValidationLoss", np.asarray(val_errors).mean(), epoch)
        accuracies = get_accuracy(network, val_dataloader)
        writer.add_scalar("Accuracy", np.asarray(accuracies).mean(), epoch)


if __name__ == "__main__":
    train(filepath_training_data=config.filepath_full_control_2w, epochs=config.epochs)
