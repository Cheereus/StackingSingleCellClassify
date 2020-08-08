'''
Description: 
Author: CheeReus_11
Date: 2020-08-08 16:38:35
LastEditTime: 2020-08-08 18:09:12
LastEditors: CheeReus_11
'''
import numpy as np
from sklearn.manifold import TSNE

# t-SNE
def t_SNE(data, dim=2):
    data = np.array(data)
    tsne = TSNE(n_components=dim, init='pca')
    tsne.fit_transform(data)
    return tsne.embedding_

