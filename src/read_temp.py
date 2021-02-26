import joblib
import numpy as np
from ReadData import read_from_csv, read_from_txt

# data = read_from_csv('data/GSE86469.csv')
data = read_from_txt('data/GSE57872_GBM_data_matrix.txt')
# print(data)
labels = data[0][:430]
labels = [i.split('_')[0] for i in labels]
print(labels)

data = [i[1:431] for i in data[1:]]
data = np.array(data)
print(data)
print(data.shape)
joblib.dump(data.T.astype(np.float), 'datasets/GSE57872.pkl')
joblib.dump(labels, 'datasets/GSE57872_labels.pkl')
