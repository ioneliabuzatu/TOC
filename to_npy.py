""" To convert csv to npy for faster loading. This will output a final npy file with the selected files concatenated."""
import os

import numpy as np
import pandas as pd

import config


def sample_type_name(csv_filename):
    if "control" in csv_filename:
        return "control"
    elif "12h" in csv_filename:
        return "12h_after_crush"
    elif "1d" in csv_filename:
        return "1d_after_crush"
    elif "2d" in csv_filename:
        return "2d_after_crush"
    elif "4d" in csv_filename:
        return "4d_after_crush"
    elif "1w" in csv_filename:
        return "1w_after_crush"
    elif "2w" in csv_filename:
        return "2w_after_crush"


def bytesto(bytes, to, bsize=1024):
    a = {'k': 1, 'm': 2, 'g': 3, 't': 4, 'p': 5, 'e': 6}
    r = float(bytes)
    return bytes / (bsize ** a[to])


def csv_to_npy():
    """dtype=object is used to have mixed types numpy array"""
    for filename in os.listdir(config.PATH_TO_UNZIPPED_CSV_FILES):
        filepath = os.path.join(config.PATH_TO_UNZIPPED_CSV_FILES, filename)
        print(f"Converting {filepath} to npy...")
        pandas_chunk_df = pd.read_csv(filepath)
        n_rows, n_cols = pandas_chunk_df.shape
        pandas_chunk_df.loc[n_rows] = ["sample_type"] + [sample_type_name(filename)] * (pandas_chunk_df.shape[1] - 1)
        df_to_numpy = pandas_chunk_df.to_numpy(dtype=object)
        np.save(f"Tran_RGC_scRNA/npys/processed-data/{filename}.npy", df_to_numpy)


def concatenate_npy(npys_path=config.root_single_processed_npy_files, memory_limit=True):
    all_rna = None
    for idx_file, npy_filename in enumerate(os.listdir(npys_path)):
        if not os.path.isfile(os.path.join(npys_path, npy_filename)):
            continue
        if memory_limit:
            if "12h" in npy_filename or "1d" in npy_filename or "2d" in npy_filename:
                continue
        print(f"Loading file #{idx_file}: {npy_filename} out of {len(os.listdir(npys_path))} files")
        filepath = os.path.join(npys_path, npy_filename)
        load_file = np.load(filepath, allow_pickle=True).T
        data_removed_header_sample_type = load_file[1:, :]

        if all_rna is not None:
            all_rna = np.vstack([all_rna, data_removed_header_sample_type])
        else:
            all_rna = np.vstack([load_file[0, :], data_removed_header_sample_type])

        del load_file
        del data_removed_header_sample_type

        bytes = all_rna.size * all_rna.itemsize
        print(f"size {bytesto(bytes, 'g'):.3f} GB")

    np.save(f"{config.root_npys}/stacked/control_4d_1w_2w.npy", all_rna)


if __name__ == "__main__":
    # csv_to_npy()
    concatenate_npy()
