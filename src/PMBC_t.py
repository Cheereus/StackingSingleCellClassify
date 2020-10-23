import xlrd
from Utils import get_color, draw_scatter

# 文件路径
filePath = 'data/41592_2017_BFnmeth4179_MOESM235_ESM.xlsx'
x1 = xlrd.open_workbook(filePath)

sheet = x1.sheets()

x = sheet[0].col_values(1)[1:]
y = sheet[0].col_values(2)[1:]
labels = sheet[0].col_values(3)[1:]
print(labels)

# get color list based on labels
default_colors = ['b', 'g', 'r', 'm', 'y', 'k']
colors = get_color(labels, default_colors)

# plot
draw_scatter(x, y, labels, colors)
