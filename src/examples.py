'''
Description: 
Author: CheeReus_11
Date: 2020-08-08 17:17:57
LastEditTime: 2020-08-09 08:34:10
LastEditors: CheeReus_11
'''
import matplotlib.pyplot as plt
from ReadData import read_from_mat
from DimensionReduction import t_SNE, get_pca
from Utils import get_color

# read data
X = read_from_mat('data/corr/A_mouse.mat')['A']

# dimenison reduction 
# t-SNE
dim_data = t_SNE(X, perp=5)

# PCA
# dim_data, ratio, result = get_pca(X, c=2, with_normalize=True)

# get two coordinates
x = [i[0] for i in dim_data]
y = [i[1] for i in dim_data]

# read labels
labels = read_from_mat('data/corr/Labels_mouse.mat')['Labels']
labels = [i[0][0] for i in labels]
print(labels)

# get color list based on labels
colors = get_color(labels)
print(colors)

# plot
plt.figure(figsize=(15,15))
plt.scatter(x, y, c=colors)
plt.show()
