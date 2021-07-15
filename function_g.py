import torch
from torch import nn


class Controller(nn.Module):
    """ Function to change input into its opposite state {healthy, unhealthy} """

    def __init__(self, num_genes: int):
        """
        Parameters
        ----------
        num_classes : int
            The number of output classes in the data.
        """
        super(Controller, self).__init__()
        self.fc1 = nn.Linear(num_genes, num_genes * 2)
        self.fc2 = nn.Linear(num_genes * 2, num_genes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
