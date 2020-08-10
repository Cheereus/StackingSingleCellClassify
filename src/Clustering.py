'''
Description: 
Author: CheeReus_11
Date: 2020-08-10 07:33:24
LastEditTime: 2020-08-10 16:38:45
LastEditors: CheeReus_11
'''
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

def k_means(X, k):
    k_m_model = KMeans(n_clusters=k, max_iter=300, n_init=40, init='k-means++', n_jobs=-1)
    k_m_model.fit(X)
    return k_m_model.labels_.tolist()

def knn(X, y, k):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X, y)
    return knn_model

def hca(X, k=None):
    hca_model = linkage(X, 'ward')
    return hca_model

# dendogram for hca
def hca_dendrogram(model):
    plt.figure(figsize=(50, 10))
    dendrogram(model, leaf_rotation=90., leaf_font_size=8)
    plt.show()

# labels of hca
def hca_labels(model, n_clusters):
    labels = fcluster(model, n_clusters, criterion='maxclust')
    return labels