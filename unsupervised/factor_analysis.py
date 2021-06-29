import numpy as np
from sklearn.decomposition import FactorAnalysis

toy_data = "../Tran_RGC_scRNA/npys/toy-data/control_12h_1d_stacked_toy.npy"
load_toy_data = np.load(toy_data, allow_pickle=True)
drop_categorical_column_of_sample_type = load_toy_data[:,:-1]

transformer = FactorAnalysis(n_components=7, random_state=0)
X_transformed = transformer.fit_transform(drop_categorical_column_of_sample_type)
print(X_transformed.shape)

print()
