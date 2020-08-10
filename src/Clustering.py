'''
Description: 
Author: CheeReus_11
Date: 2020-08-10 07:33:24
LastEditTime: 2020-08-10 08:19:28
LastEditors: CheeReus_11
'''
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier  


def k_means(X, k):
    k_m_model = KMeans(n_clusters=k, max_iter=300, n_init=40, init='k-means++', n_jobs=-1)
    k_m_model.fit(X)
    return k_m_model.labels_.tolist()

def knn(X, y, k):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X, y)
    return knn_model
