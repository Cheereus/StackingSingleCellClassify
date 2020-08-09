'''
Description: 
Author: CheeReus_11
Date: 2020-08-08 17:42:59
LastEditTime: 2020-08-09 14:45:29
LastEditors: CheeReus_11
'''

from collections import Counter

# default_colors = ['c', 'b', 'g', 'r', 'm', 'y', 'k']
# default_colors = [[1, 0, 0],[0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0.5, 0.5, 0.5]]
default_colors = [[0, 0.8, 1],[0, 0.5, 0.5], [0.2, 0.8, 0.8], [0.2, 0.4, 1], [0.6, 0.8, 1], [1, 0.6, 0.8], [0.8, 0.6, 1], [1, 0.8, 0.6]]

# Get color list for drawing based on labels, more color values can also be customized
def get_color(labels, colors=None):
    len_labels = len(labels)
    t = 0
    c = [None] * len_labels
    count_result = Counter(labels)
    colors = colors or default_colors
    # colors = colors or range(1, len(count_result.keys()) + 1)
    
    for i in count_result.keys():
        print(i,colors[t])
        for j in range(len_labels):
            if i == labels[j]:
                c[j] = colors[t]
        t += 1
    
    return c
