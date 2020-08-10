'''
Description: 
Author: CheeReus_11
Date: 2020-08-08 17:17:57
LastEditTime: 2020-08-10 11:03:10
LastEditors: CheeReus_11
'''
import matplotlib.pyplot as plt
from ReadData import read_from_mat
from DimensionReduction import t_SNE, get_pca
from Utils import get_color, draw_scatter, accuracy
from Clustering import k_means, knn

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
# print(labels)

# knn training and predict
model = knn(dim_data, labels, 3)
labels_predict = model.predict(dim_data)

print('Accuracy:', accuracy(labels_predict, labels))

# get color list based on labels
default_colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k']
colors = get_color(labels_predict, default_colors)

# draw
draw_scatter(x, y, labels_predict, colors)
