'''
Description: 
Author: CheeReus_11
Date: 2020-08-08 17:42:59
LastEditTime: 2020-08-08 18:05:22
LastEditors: CheeReus_11
'''

from collections import Counter

# default colors
default_color = ['c', 'b', 'g', 'r', 'm', 'k']
# Get color list for drawing based on labels, more color values can also be customized
def get_color(labels, colors=default_color):

    t = 0
    c = []
    count_result = Counter(labels)
    for i in count_result.keys():
        for j in labels:
            if i == j:
                c.append(colors[t])
        t += 1
    return c
