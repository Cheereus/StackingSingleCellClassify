'''
Description: 
Author: CheeReus_11
Date: 2020-08-08 17:42:59
LastEditTime: 2020-08-09 08:06:01
LastEditors: CheeReus_11
'''

from collections import Counter

# Get color list for drawing based on labels, more color values can also be customized
def get_color(labels, colors=None):
    t = 0
    c = []
    count_result = Counter(labels)
    colors = colors or range(1, len(count_result.keys()) + 1)
    
    for i in count_result.keys():
        for j in labels:
            if i == j:
                c.append(colors[t])
        t += 1
    
    return c
