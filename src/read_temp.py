import joblib
import numpy as np
from ReadData import read_from_csv, read_from_txt

data = read_from_txt('data/GSE60361.txt')

labels = data[1][3:]
# labels = ['d0' for _ in range(data.shape[0]-1)]
# print(len(labels))
# #
# data = data[1:]
data_list = []
for i in range(10, data.shape[0]):
    data_list.append(data[i][2:])

data_list = np.array(data_list)
print(data_list.shape)
joblib.dump(data_list.astype(np.float32).T, 'datasets/GSE60361.pkl')
joblib.dump(labels, 'datasets/GSE60361_labels.pkl')
