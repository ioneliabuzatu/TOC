from dask import dataframe as dd

if __name__ == "__main__":
    control_count_matrix_filename = '/Tran_RGC_scRNA/unzipped/GSE137398_ONCRGCs_control_count_mat.csv'

    dask_df = dd.read_csv(control_count_matrix_filename, sample=1000000)
    print(dask_df.head())
    print()
