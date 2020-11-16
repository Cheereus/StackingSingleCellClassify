import joblib
from ReadData import read_from_csv
import numpy as np

data = np.array(read_from_csv('data/Chu_cell_time.csv'))
print(data.shape)
labels = [i.split('_')[0] for i in data[0][1:]]
genes = [i for i in data.T[0][1:]]
print(len(genes), len(labels))
print(data[1:, 1:].shape)

joblib.dump(data[1:, 1:].T.astype(np.float64), 'datasets/Chu_cell_time.pkl')
joblib.dump(labels, 'datasets/Chu_cell_time_labels.pkl')
joblib.dump(genes, 'datasets/Chu_cell_time_genes.pkl')
