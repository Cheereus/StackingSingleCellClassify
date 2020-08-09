'''
Description: 
Author: CheeReus_11
Date: 2020-08-08 16:38:35
LastEditTime: 2020-08-09 08:25:29
LastEditors: CheeReus_11
'''
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA  
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

# t-SNE
def t_SNE(data, dim=2, perp=30):
    data = np.array(data)
    tsne = TSNE(n_components=dim, init='pca', perplexity=perp)
    tsne.fit_transform(data)
    return tsne.embedding_

def get_pca(X, c=3, with_normalize=False):

    if with_normalize:
        prepress = Normalizer()
        X = prepress.fit_transform(X)
        
    pca_result = PCA(n_components=c)
    pca_result.fit(X)
    newX = pca_result.fit_transform(X)

    return newX, pca_result.explained_variance_ratio_, pca_result
