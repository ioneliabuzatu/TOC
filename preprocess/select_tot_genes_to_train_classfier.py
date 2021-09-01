import numpy as np


def subset_genes_count_matrix_to_train_classifier(filepath: str, save_to_filepath: str):
    """
    Selects the genes columns of interest and saves the new count matrix to an npy file.
    :filepath str, the original count matrix with all genes.
    :save_to_filepath str, where to save the selected genes count matrix.
    """
    control_diseased_stacked = np.load(filepath, allow_pickle=True)
    assert type(control_diseased_stacked[0][0]) == str, print("We expect the first row to be the genes names.")
    genes_to_select = {
        "Ank2": 0, "Cartpt": 1, "Chrm2": 2, "Eomes": 3, "Kctd4": 4, "Mafb": 5,
        "Neurod2": 6, "Opn4": 7, "Pou4f3": 8, "Ttn": 9, "mt-Cytb": 10, "mt-Nd4": 11
    }

    data_to_save = np.zeros((control_diseased_stacked.shape[0]-1, len(genes_to_select)+1), dtype=object)
    assert len(data_to_save) == len(control_diseased_stacked)-1

    for idx, gene_name in enumerate(control_diseased_stacked[0]):
        if gene_name in genes_to_select.keys():
            data_to_save[:, genes_to_select[gene_name]] = control_diseased_stacked[1:, idx]

    vector_labels = control_diseased_stacked[1:, -1]
    data_to_save[:, -1] = vector_labels

    np.save(save_to_filepath, data_to_save)
    print(f"\n *** Expressions of selected genes saved as `{save_to_filepath}` *** \n")


if __name__ == "__main__":
    subset_genes_count_matrix_to_train_classifier(
        filepath="../data/Tran_RGC_scRNA/npys/stacked/control_2w.npy",
        save_to_filepath="../control_disease_12_genes_expressions.npy"
    )