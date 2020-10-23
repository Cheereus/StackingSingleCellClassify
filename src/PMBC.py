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

# joblib.dump(X, 'datasets/human_islets.pkl')
# joblib.dump(labels, 'datasets/human_islets_labels.pkl')

print(labels)
print(X.shape)
print(datetime.datetime.now())

# t-SNE
dim_data = t_SNE(X, perp=50, with_normalize=True)
# PCA
# dim_data, ratio, result = get_pca(X, c=2, with_normalize=True)
# get two coordinates
x = [i[0] for i in dim_data]
y = [i[1] for i in dim_data]

# get color list based on labels
default_colors = ['b', 'g', 'r', 'm', 'y', 'k']
colors = get_color(labels, default_colors)

# plot
draw_scatter(x, y, labels, colors)

