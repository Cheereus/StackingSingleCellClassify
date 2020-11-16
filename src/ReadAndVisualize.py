import joblib
from DimensionReduction import t_SNE, get_pca
from Utils import get_color, draw_scatter

data = joblib.load('ae_output/ae_dim_data_99.pkl')
print(data.shape)
labels = joblib.load('ae_output/labels.pkl')
print(len(labels))

dim_data = t_SNE(data, perp=40, with_normalize=True)

x = [i[0] for i in dim_data]
y = [i[1] for i in dim_data]
# get color list based on labels
default_colors = ['c', 'b', 'g', 'r', 'm', 'y', 'c', 'k']
colors = get_color(labels, default_colors)

# plot
draw_scatter(x, y, labels, colors)
