from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import mutual_info_score
import numpy as np
import math
import datetime


# 欧式距离相似度函数，输入矩阵以行为样本
def SimDistance(X):

    euc_dis = euclidean_distances(X)
    sim_dis = [1 / (1 + i) for i in euc_dis]
    # print('SimDistance computed', datetime.datetime.now())
    return np.array(sim_dis)


# 相关系数相似度函数，输入矩阵以行为样本
def SimCorrelation(X):

    corr = np.corrcoef(X)
    sim_corr = [(1 + i) / 2 for i in corr]
    # print('SimCorrelation computed', datetime.datetime.now())
    return np.array(sim_corr)


# 互信息相似度函数，输入矩阵以行为样本
def SimMutual(X):

    n_sample, n_feature = X.shape
    sim_mutual = np.zeros((n_sample, n_sample))
    max_mi = 0

    for i in range(n_sample):
        for j in range(i, n_sample):
            mu = 0
            if i != j:
                c1 = np.cov(X[i])
                c2 = np.cov(X[j])
                c3 = np.linalg.det(np.cov(X[i], X[j]))
                if c3 < 0:
                    c3 = -c3
                mu = 0.5 * math.log(c1 * c2 / c3)
                if mu > max_mi:
                    max_mi = mu
            sim_mutual[i][j] = mu
            sim_mutual[j][i] = mu

    print(sim_mutual)
    sim_mutual = sim_mutual / max_mi
    for i in range(n_sample):
        sim_mutual[i][i] = 1

    # print('SimMutual computed', datetime.datetime.now())
    return sim_mutual


# 整合的相似性度量函数
def Similarity(X, alpha, beta, gamma=None):

    if gamma is None:
        gamma = 1 - alpha - beta
    # print('Similarity computed', datetime.datetime.now())
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

