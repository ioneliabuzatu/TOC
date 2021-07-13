import config
from classfier_cell_state import CellStateClassifier
import numpy as np


def cost_function(classifier, cell_rna_content, true_label):
    return classifier(cell_rna_content) == true_label


def main():
    for time_step in range(10):
        dummy_input = np.random.randint(10, 90, 4)
        network = CellStateClassifier(num_genes=4)
        cost = cost_function(network, dummy_input, true_label=1)

main()