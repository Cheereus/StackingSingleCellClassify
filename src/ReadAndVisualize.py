import joblib
from DimensionReduction import t_SNE, get_pca
from Utils import get_color, draw_scatter

data = joblib.load('datasets/Chu_cell_time.pkl')
labels = joblib.load('datasets/Chu_cell_time_labels.pkl')

dim_data, ratio, result = get_pca(data, c=20, with_normalize=True)
dim_data = t_SNE(dim_data, perp=40, with_normalize=True)

x = [i[0] for i in dim_data]
y = [i[1] for i in dim_data]
# get color list based on labels
default_colors = ['b', 'g', 'r', 'm', 'y', 'c']
colors = get_color(labels, default_colors)

# plot
draw_scatter(x, y, labels, colors)
