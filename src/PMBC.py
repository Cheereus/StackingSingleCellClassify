from ReadData import read_from_txt
from DimensionReduction import t_SNE, get_pca
from Utils import get_color, draw_scatter
import numpy as np
import joblib
import datetime

dataset = read_from_txt('data/GSM2486333_PBMC.txt')
labels = dataset[0][:3694]
labels = [i.split('_')[0].split('"')[1] for i in labels]
X = []
for line in dataset[1:]:
    X.append(line[1:])
X = np.array(X).T.astype(np.float64)[:3694]

# joblib.dump(X, 'datasets/PBMC.pkl')
# joblib.dump(labels, 'datasets/PBMC_labels.pkl')

print(labels)
print(X.shape)
print(datetime.datetime.now())

# PCA
dim_data, ratio, result = get_pca(X, c=20, with_normalize=False)
print(sum(ratio))
# t-SNE
dim_data = t_SNE(dim_data, perp=40, with_normalize=False)
# get two coordinates
x = [i[0] for i in dim_data]
y = [i[1] for i in dim_data]

# get color list based on labels
default_colors = ['b', 'g', 'r', 'm', 'y', 'k']
colors = get_color(labels, default_colors)

# plot
draw_scatter(x, y, labels, colors)

