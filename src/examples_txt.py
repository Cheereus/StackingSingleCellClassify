'''
Description: 
Author: CheeReus_11
Date: 2020-08-08 17:17:57
LastEditTime: 2020-08-09 14:58:49
LastEditors: CheeReus_11
'''
import matplotlib.pyplot as plt
from ReadData import read_from_txt
from DimensionReduction import t_SNE, get_pca
from Utils import get_color
import numpy as np

# read labels
labels = read_from_txt('data/human_islet_labels.txt')
labels = [i[-1] for i in labels][1:]
print(len(labels))

# read data
X = read_from_txt('data/human_islets.txt')
X = X.T[1:, 1:].astype(np.float64)
print(X.shape)

# dimenison reduction 
# t-SNE
dim_data = t_SNE(X, perp=50, with_normalize=True)

# PCA
# dim_data, ratio, result = get_pca(X, c=2, with_normalize=True)

# get two coordinates
x = [i[0] for i in dim_data]
y = [i[1] for i in dim_data]

# get color list based on labels
colors = get_color(labels)

# plot
plt.figure(figsize=(15,15))
plt.scatter(x, y, c=colors, marker='o', linewidths=1)
plt.show()
