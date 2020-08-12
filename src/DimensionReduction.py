'''
Description: 
Author: CheeReus_11
Date: 2020-08-08 16:38:35
LastEditTime: 2020-08-12 10:25:42
LastEditors: CheeReus_11
'''
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA  
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

# Normalize
def get_normalize(data):
    prepress = Normalizer()
    X = prepress.fit_transform(data)
    return X
    
# t-SNE
def t_SNE(data, dim=2, perp=30, with_normalize=False):
    
    if with_normalize:
        data = get_normalize(data)

    data = np.array(data)
    tsne = TSNE(n_components=dim, init='pca', perplexity=perp, method='exact')
    tsne.fit_transform(data)
    
    return tsne.embedding_

# PCA
def get_pca(data, c=3, with_normalize=False):

    if with_normalize:
        X = get_normalize(data)
        
    pca_result = PCA(n_components=c)
    pca_result.fit(data)
    newX = pca_result.fit_transform(data)

    return newX, pca_result.explained_variance_ratio_, pca_result
