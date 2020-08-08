'''
Description: 
Author: CheeReus_11
Date: 2020-08-08 17:17:57
LastEditTime: 2020-08-08 18:02:04
LastEditors: CheeReus_11
'''
import matplotlib.pyplot as plt
from ReadData import read_from_mat
from DimensionReduction import t_SNE
from utils import get_color

# 读取数据并分开两个坐标
X = read_from_mat('data/corr/A_mouse.mat')['A']
dim_data = t_SNE(X)
x = [i[0] for i in dim_data]
y = [i[1] for i in dim_data]

# 读取标签并转化为颜色
labels = read_from_mat('data/corr/Labels_mouse.mat')['Labels']
labels = [int(i[0][0][0]) for i in labels]
colors = get_color(labels)

plt.scatter(x, y, c=colors)
plt.show()
