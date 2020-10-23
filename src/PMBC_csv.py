from ReadData import read_from_txt, read_from_csv
from DimensionReduction import t_SNE, get_pca
from Utils import get_color, draw_scatter
import numpy as np
import joblib
import datetime
import xlrd

dataset = read_from_txt('data/GSM2486333_PBMC.txt')
X = read_from_csv('data/PBMC.csv')
X = np.array(X).T.astype(np.float64)

# 文件路径
filePath = 'data/41592_2017_BFnmeth4179_MOESM235_ESM.xlsx'
x1 = xlrd.open_workbook(filePath)
sheet = x1.sheets()
labels = sheet[0].col_values(3)[1:]

# joblib.dump(X, 'datasets/human_islets.pkl')
# joblib.dump(labels, 'datasets/human_islets_labels.pkl')

print(labels)
print(X.shape)
print(datetime.datetime.now())

# PCA
dim_data, ratio, result = get_pca(X, c=11, with_normalize=False)
print(sum(ratio))
# t-SNE
dim_data = t_SNE(dim_data, perp=5, with_normalize=False)
# get two coordinates
x = [i[0] for i in dim_data]
y = [i[1] for i in dim_data]

# get color list based on labels
default_colors = ['b', 'g', 'r', 'm', 'y', 'k']
colors = get_color(labels, default_colors)

# plot
draw_scatter(x, y, labels, colors)

