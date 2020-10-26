import joblib
from Utils import get_color, draw_scatter
from DimensionReduction import t_SNE, get_pca
import numpy as np
import datetime
from Clustering import k_means
from Metrics import ARI, NMI

X = joblib.load('ae_output/ae_dim_data_99.pkl')
labels = joblib.load('ae_output/labels.pkl')


print(labels)
print(X.shape)
print(datetime.datetime.now())
# PCA
# dim_data, ratio, result = get_pca(X, c=11, with_normalize=False)
# print(sum(ratio))
# t-SNE
dim_data = t_SNE(X, perp=40, with_normalize=False)
print(datetime.datetime.now())
# get two coordinates
x = [i[0] for i in dim_data]
y = [i[1] for i in dim_data]

# get color list based on labels
default_colors = ['b', 'g', 'r', 'm', 'y', 'c']
colors = get_color(labels, default_colors)

# plot
draw_scatter(x, y, labels, colors)

predict_labels = k_means(X, k=6)

print(ARI(labels, predict_labels))
print(NMI(labels, predict_labels))



