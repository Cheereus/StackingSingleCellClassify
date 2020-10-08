from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mutual_info_score
import numpy as np


# 欧式距离相似度函数，输入矩阵以行为样本
def SimDistance(X):
    euc_dis = euclidean_distances(X)
    sim_dis = [1 / (1 + i) for i in euc_dis]
    return sim_dis


# 相关系数相似度函数，输入矩阵以行为样本
def SimCorrelation(X):
    corr = np.corrcoef(X)
    sim_corr = [(1 + i) / 2 for i in corr]
    return sim_corr


# 互信息相似度函数，输入矩阵以行为样本
def SimMutual(X):

    n_sample, n_feature = X.shape
    sim_mutual = np.zeros((n_sample, n_sample))

    for i in range(n_sample):
        for j in range(n_sample):
            sim_mutual[i][j] = mutual_info_score(X[i], X[j])
            sim_mutual[j][i] = mutual_info_score(X[i], X[j])

    return sim_mutual
