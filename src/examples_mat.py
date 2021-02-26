import matplotlib.pyplot as plt
from ReadData import read_from_mat
from DimensionReduction import t_SNE, get_pca
from Utils import get_color, draw_scatter
from Clustering import k_means
import joblib

# read data
X = read_from_mat('data/corr/A_islet.mat')['A']

# dimenison reduction 
# t-SNE
dim_data = t_SNE(X, perp=5, with_normalize=True)

# PCA
# dim_data, ratio, result = get_pca(X, c=2, with_normalize=True)

# get two coordinates
x = [i[0] for i in dim_data]
y = [i[1] for i in dim_data]

# read labels
labels = read_from_mat('data/corr/Labels_islet.mat')['Labels']
labels = [i[0][0] for i in labels]
print(X.shape)

joblib.dump(X, 'datasets/human_islet.pkl')
joblib.dump(labels, 'datasets/human_islet_labels.pkl')

# get color list based on labels
colors = get_color(labels)
print(colors)

# draw
draw_scatter(x, y, labels, colors)
