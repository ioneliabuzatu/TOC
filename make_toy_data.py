import numpy as np

import config


def take_random_rows_xor_cols_indices(matrix_shape: tuple, select_n_rows: int, select_n_cols: int,
                                      take_only_rows_idx: bool = False,
                                      take_only_cols_idx: bool = False):
    """Selects random cols and rows and ignore first row and last column"""
    n_rows, n_cols = matrix_shape
    idx_rows = np.random.randint(1, n_rows, size=select_n_rows)
    if take_only_rows_idx and not take_only_cols_idx:
        return idx_rows
    idx_cols = np.random.randint(0, n_cols - 1, size=select_n_cols)
    if take_only_cols_idx and not take_only_rows_idx:
        return idx_cols
    if take_only_rows_idx and take_only_cols_idx:
        return idx_rows, idx_cols
    return NotImplementedError("take_only_rows_idx and take_only_cols_idx should not be both true")


def make_toy_data(select_n_rows, select_n_cols, labels: list, save=True):
    all_rna_all_samples = np.load(config.all_rna_data_npy_filepath, allow_pickle=True)

    extra_row_for_header = 1
    extra_column_for_labels = 1
    toy_data = np.zeros(
        (select_n_rows * len(labels) + extra_row_for_header, select_n_cols + extra_column_for_labels), dtype=object
    )
    idx_cols_genes = take_random_rows_xor_cols_indices(
        all_rna_all_samples.shape, select_n_rows, select_n_cols, take_only_cols_idx=True
    )

    toy_data[0, :-1] = all_rna_all_samples[0, :][idx_cols_genes]
    toy_data[0, -1] = "sample_type"

    for sample_idx, sample_name in enumerate(labels):
        mask_sample = all_rna_all_samples[:, -1] == sample_name

        select_sample_rows = all_rna_all_samples[mask_sample, :]

        idx_rows_sample = take_random_rows_xor_cols_indices(
            select_sample_rows.shape, select_n_rows, select_n_cols, take_only_rows_idx=True
        )

        toy_portion = select_sample_rows[idx_rows_sample, :][:, idx_cols_genes]

        toy_data[1 + (sample_idx * select_n_rows):(sample_idx * select_n_rows) + select_n_rows+1, :-1] = toy_portion
        toy_data[1 + (sample_idx * select_n_rows):(sample_idx * select_n_rows) + select_n_rows+1, -1] = sample_name

    if save:
        np.save(f"{config.root_toy_data}/{'_and_'.join(labels)}_{select_n_rows}x{select_n_cols}_toy.npy", toy_data)

    return toy_data


if __name__ == "__main__":
    labels = ["control", "2w_after_crush"]
    make_toy_data(select_n_rows=2000, select_n_cols=2000, labels=labels, save=True)
