""" To convert csv to npy for faster loading. This will output a final npy file with the selected files concatenated."""
import os

import numpy as np
import pandas as pd
import config


def sample_type_name(csv_filename):
    if "control" in csv_filename:
        return "control"
    elif "1d" in csv_filename:
        return "1d_after_crush"
    elif "12h" in csv_filename:
        return "2d_after_crush"


def csv_to_npy():
    """dtype=object is used to have mixed types numpy array"""
    PATH_TO_UNZIPPED_CSV_FILES = "Tran_RGC_scRNA/unzipped"
    for filename in os.listdir(PATH_TO_UNZIPPED_CSV_FILES):
        filepath = os.path.join(PATH_TO_UNZIPPED_CSV_FILES, filename)
        print(f"Converting {filepath} to npy...")
        pandas_chunk_df = pd.read_csv(filepath)
        n_rows, n_cols = pandas_chunk_df.shape
        pandas_chunk_df.loc[n_rows] = ["sample_type"] + [sample_type_name(filename)] * (pandas_chunk_df.shape[1] - 1)
        df_to_numpy = pandas_chunk_df.to_numpy(dtype=object)
        np.save(f"Tran_RGC_scRNA/npys/{filename}.npy", df_to_numpy)


def concatenate_npy(npys_path="Tran_RGC_scRNA/npys"):
    all_rna = None
    toy_rna = None
    for npy_filename in os.listdir(npys_path):
        if not os.path.isfile(os.path.join(npys_path, npy_filename)):
            continue
        filepath = os.path.join(npys_path, npy_filename)
        load_file = np.load(filepath, allow_pickle=True)
        transposed = load_file.T

        n_cells, n_genes = transposed.shape
        idx = np.random.randint(n_cells - 1, size=1000)
        toy_data_portion = transposed[idx, :]
        np.save(f"{npys_path}/toy-data/{npy_filename}.toy.npy", toy_data_portion)

        if all_rna is not None:
            all_rna = np.vstack([all_rna, transposed])
        else:
            all_rna = transposed

        if toy_rna is not None:
            toy_rna = np.vstack([toy_rna, toy_data_portion])
        else:
            toy_rna = toy_data_portion

    np.save(f"{npys_path}/processed-data/control_12h_1d_stacked.npy", all_rna)
    np.save(f"{npys_path}/toy-data/control_12h_1d_stacked_toy.npy", toy_rna)


def get_toy_data():
    all_rna_all_samples = np.load(config.all_rna_data_npy_filepath, allow_pickle=True)
    print()


if __name__ == "__main__":
    # csv_to_npy()
    # concatenate_npy()
    get_toy_data()