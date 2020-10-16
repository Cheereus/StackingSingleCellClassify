from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mutual_info_score
import numpy as np


# 欧式距离相似度函数，输入矩阵以行为样本
def SimDistance(X):

    euc_dis = euclidean_distances(X)
    sim_dis = [1 / (1 + i) for i in euc_dis]
    return np.array(sim_dis)


# 相关系数相似度函数，输入矩阵以行为样本
def SimCorrelation(X):

    corr = np.corrcoef(X)
    sim_corr = [(1 + i) / 2 for i in corr]
    return np.array(sim_corr)


# 互信息相似度函数，输入矩阵以行为样本
def SimMutual(X):

    n_sample, n_feature = X.shape
    sim_mutual = np.zeros((n_sample, n_sample))

    for i in range(n_sample):
        for j in range(n_sample):
            sim_mutual[i][j] = mutual_info_score(X[i], X[j])
            sim_mutual[j][i] = mutual_info_score(X[i], X[j])

    return sim_mutual


# 整合的相似性度量函数
def Similarity(X, alpha, beta, gamma):

    return alpha * SimDistance(X) + beta * SimCorrelation(X) + gamma * SimMutual(X)


# 细胞关联矩阵
def RelevanceMatrix(labels):

    n_samples = len(labels)
    rm = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if labels[i] == labels[j]:
                rm[i, j] = 1
                rm[j, i] = 1

    return rm


# 一致关联矩阵
def ConsistentMatrix(relevance, m):

    S = np.zeros((m, m))
    for rel in relevance:
        S = S + rel
    S = np.ones((m, m)) - S / m
    return S
