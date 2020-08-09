'''
Description: 
Author: CheeReus_11
Date: 2020-08-08 17:17:57
LastEditTime: 2020-08-09 10:01:39
LastEditors: CheeReus_11
'''
import matplotlib.pyplot as plt
from ReadData import read_from_csv
from DimensionReduction import t_SNE, get_pca
from Utils import get_color
import numpy as np

# read data
data = read_from_csv('data/yang_human_embryo.csv')

X = data.T[:90,1:]
print(X.shape)

labels = data.T[:90,0]
labels = [i.partition('#')[0] for i in labels]
print(labels)

# dimenison reduction 
# t-SNE
dim_data = t_SNE(X, perp=40, with_normalize=True)

# PCA
# dim_data, ratio, result = get_pca(X, c=2, with_normalize=True)
# print(ratio)

# get two coordinates
x = [i[0] for i in dim_data]
y = [i[1] for i in dim_data]

# get color list based on labels
default_colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k']
colors = get_color(labels,default_colors)
print(colors)

# plot
plt.figure(figsize=(15,15))
plt.scatter(x, y, c=colors)
plt.show()
