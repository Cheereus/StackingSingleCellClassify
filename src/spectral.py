from sklearn.cluster import SpectralClustering

from ReadData import read_from_mat
from DimensionReduction import t_SNE, get_pca
from Utils import get_color, draw_scatter
import numpy as np

# read data
X = read_from_mat('data/corr/A_islet.mat')['A']

print(X.shape)

# read labels
labels = read_from_mat('data/corr/Labels_islet.mat')['Labels']
labels = [i[0][0] for i in labels]

# dimenison reduction
# t-SNE
dim_data = t_SNE(X, perp=5, with_normalize=True)

# PCA
# dim_data, ratio, result = get_pca(X, c=2, with_normalize=True)
# print(ratio)

# get two coordinates
x = [i[0] for i in dim_data]
y = [i[1] for i in dim_data]
#
labels_predict = SpectralClustering(n_clusters=6, affinity='nearest_neighbors').fit_predict(X)
print(labels_predict)

# get color list based on labels
default_colors = ['c', 'b', 'g', 'r', 'm', 'y']
colors = get_color(labels_predict)

# plot
draw_scatter(x, y, labels_predict, colors)
