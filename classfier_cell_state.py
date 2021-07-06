import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor, Resize
from torch.utils.data import Subset


@torch.no_grad()
def evaluate(network: nn.Module, data: DataLoader, metric) -> list:
    network.eval()
    device = next(network.parameters()).device

    errors = []
    for x, y in data:
        x, y = x.to(device), y.to(device)
        logits = network(x)
        if logits.dim != y.dim:
            y = y.unsqueeze(1)
        errors.append(metric(logits, y).item())

    return errors


@torch.enable_grad()
def update(network: nn.Module, data: DataLoader, loss: nn.Module,
           opt: optim.Optimizer, regulariser: nn.Module = None) -> list:
    network.train()
    device = next(network.parameters()).device

    errs = []
    for x, y in data:
        x, y = x.to(device), y.to(device)
        logits = network(x)
        if logits.dim != y.dim:
            y = y.unsqueeze(1)
        err = loss(logits, y)
        errs.append(err.item())

        opt.zero_grad()
        err.backward()
        opt.step()

    return errs


@torch.no_grad()
def accuracy(logits, targets):
    """
    Compute the accuracy for given logits and targets.

    Parameters
    ----------
    logits : (N, K) torch.Tensor
        A mini-batch of logit vectors from the network.
    targets : (N, ) torch.Tensor
        A mini_batch of target scalars representing the labels.

    Returns
    -------
    acc : () torch.Tensor
        The accuracy over the mini-batch of samples.
    """
    tot_correct = tot_samples = 0

    # _, predictions = logits.max(1)
    predictions = logits
    tot_correct += (predictions == targets).sum()
    tot_samples += predictions.size(0)

    accuracy_samples = (tot_correct.item() / tot_samples) * 100

    return accuracy_samples


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


class CellStateClassifier(nn.Module):
    """ Classifier network for classifying cell state {healthy, unhealthy} """

    def __init__(self, num_genes: int):
        """
        Parameters
        ----------
        num_classes : int
            The number of output classes in the data.
        """
        super(CellStateClassifier, self).__init__()
        self.fc = nn.Linear(num_genes, 1)
        self.classifier = nn.Sequential(
            self.fc
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class TranscriptomicsDataset(Dataset):
    def __init__(self, filepath_data, normilize_by_max=True):
        self.normalize_data = normilize_by_max
        self.data = np.load(filepath_data, allow_pickle=True)
        if isinstance(self.data[0, 0], str):
            self.genes_names = self.data[0, :]
            self.data = self.data[1:, :]
        self.preprocess_data()
        self.labels_encoding, self.labels_categorical = np.unique(self.data[:, -1], return_inverse=True)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.array(self.data[idx, :-1], dtype=np.float32))
        y = torch.from_numpy(np.array(self.labels_categorical[idx], dtype=np.float32))
        return x, y

    def __len__(self):
        return len(self.data)

    def preprocess_data(self):
        """ TODO: try TPM normalization too """
        # x_normed = data / data.max(axis=1)
        if self.normalize_data:
            self.data[1:, :-1] = normalize(self.data[1:, :-1], axis=1, norm="max")


def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {'train': Subset(dataset, train_idx), 'val': Subset(dataset, val_idx)}
    return datasets