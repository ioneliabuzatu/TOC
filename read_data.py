import numpy as np
import scanpy as sc
from dask import dataframe as dd

sample_strings = ['control', '12h_afterCrush', '1d_afterCrush', '2d_afterCrush', '4d_afterCrush', '1w_afterCrush',
                  '2w_afterCrush']
file_base = '/home/ionelia/pycharm-projects/master/semester2/internshipISTaustria/TranscriptomicsOptimalControl' \
            '/Tran_RGC_scRNA/unzipped/GSE137398_ONCRGCs_'
data_file_end = '_count_mat.csv'


sample = sample_strings[0]
data_file = file_base + sample + data_file_end

dask_df = dd.read_csv(data_file,  sample=1000000)
print(dask_df.head())
print()