'''
Description: 
Author: CheeReus_11
Date: 2020-08-08 17:17:57
LastEditTime: 2020-08-09 17:03:58
LastEditors: CheeReus_11
'''
import matplotlib.pyplot as plt
from ReadData import read_from_txt
from DimensionReduction import t_SNE, get_pca
from Utils import get_color, draw_scatter
import numpy as np
import joblib

# read labels
labels = read_from_txt('data/human_islets_labels.txt')
labels = [i[-1] for i in labels][1:]
print(len(labels))

# read data
X = read_from_txt('data/human_islets.txt')
X = X.T[1:, 1:].astype(np.float64)
print(X.shape)
joblib.dump(X, 'datasets/human_islets.pkl')
joblib.dump(labels, 'datasets/human_islets_labels.pkl')
# dimenison reduction
# t-SNE
dim_data = t_SNE(X, perp=5, with_normalize=True)

# PCA
# dim_data, ratio, result = get_pca(X, c=2, with_normalize=True)

# get two coordinates
x = [i[0] for i in dim_data]
y = [i[1] for i in dim_data]

# get color list based on labels
colors = get_color(labels)

# plot
draw_scatter(x, y, labels, colors)
