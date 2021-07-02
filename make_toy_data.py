import numpy as np

import config


def random_rows_and_cols_idx(matrix_shape, select_n_rows, select_n_cols):
    """ignore first row and last column"""
    n_rows, n_cols = matrix_shape
    idx_rows = np.random.randint(1, n_rows, size=select_n_rows)
    idx_cols = np.random.randint(0, n_cols - 1, size=select_n_cols)
    return idx_rows, idx_cols


def make_toy_data(select_n_rows, select_n_cols, save=True):
    all_rna_all_samples = np.load(config.all_rna_data_npy_filepath, allow_pickle=True)

    mask_control = all_rna_all_samples[:, -1] == "control"
    mask_12h_after_crush = all_rna_all_samples[:, -1] == "2d_after_crush"
    mask_1d_after_crush = all_rna_all_samples[:, -1] == "1d_after_crush"

    select_control = all_rna_all_samples[mask_control, :]
    select_twelve_hours_after_crush = all_rna_all_samples[mask_12h_after_crush, :]
    select_one_day_after_crush = all_rna_all_samples[mask_1d_after_crush, :]

    idx_rows_control, idx_cols = random_rows_and_cols_idx(select_control.shape, select_n_rows, select_n_cols)
    idx_rows_twelve_hours, _ = random_rows_and_cols_idx(select_twelve_hours_after_crush.shape, select_n_rows,
                                                        select_n_cols)
    idx_rows_one_day, _ = random_rows_and_cols_idx(select_one_day_after_crush.shape, select_n_rows, select_n_cols)

    toy_control = select_control[idx_rows_control, :][:, idx_cols]
    toy_12h = select_twelve_hours_after_crush[idx_rows_twelve_hours, :][:, idx_cols]
    toy_1d = select_one_day_after_crush[idx_rows_one_day, :][:, idx_cols]

    toy_control_with_sample_col = np.zeros((toy_control.shape[0], toy_control.shape[1] + 1), dtype=object)
    toy_control_with_sample_col[:, :-1] = toy_control
    toy_control_with_sample_col[:, -1] = ["control"] * toy_control.shape[0]

    toy_12h_with_sample_col = np.zeros((toy_12h.shape[0], toy_12h.shape[1] + 1), dtype=object)
    toy_12h_with_sample_col[:, :-1] = toy_12h
    toy_12h_with_sample_col[:, -1] = ["12h_after_crush"] * toy_12h.shape[0]

    toy_1d_with_sample_col = np.zeros((toy_1d.shape[0], toy_1d.shape[1] + 1), dtype=object)
    toy_1d_with_sample_col[:, :-1] = toy_1d
    toy_1d_with_sample_col[:, -1] = ["1d_after_crush"] * toy_1d.shape[0]

    gene_names = np.zeros(len(idx_cols) + 1, dtype=object)
    gene_names[:-1] = all_rna_all_samples[0, :][idx_cols]
    gene_names[-1] = "sample_type"
    toy_rna = np.vstack([gene_names, toy_control_with_sample_col, toy_12h_with_sample_col,
                         toy_1d_with_sample_col])

    if save:
        np.save(f"{config.root_toy_data}/control_12h_1d_stacked_toy.npy", toy_rna)
    return toy_rna


if __name__ == "__main__":
    make_toy_data(select_n_rows=100, select_n_cols=1000)
