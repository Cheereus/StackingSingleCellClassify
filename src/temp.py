import joblib

from Distance import SimDistance, SimCorrelation, SimMutual, Similarity
import numpy as np
from Clustering import hca, hca_dendrogram, hca_labels, k_means
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from Metrics import ARI, accuracy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering

X = joblib.load('datasets/Xin_human_islets.pkl')
labels = joblib.load('datasets/Xin_human_islets_labels.pkl')
print(X.shape)

sim_dis = SimDistance(X)
sim_cor = SimCorrelation(X)
sim_mu = SimMutual(X)
sim_data = Similarity(X, alpha=0.8, beta=0.1, gamma=0.1)


def calc_and_output(sim):
    # labels_predict = k_means(sim, 6)
    knn_model = KNeighborsClassifier(n_neighbors=6, metric='precomputed')
    knn_model.fit(np.max(sim) - sim, labels)
    labels_predict = knn_model.predict(np.max(sim) - sim)
    print('ARI:', ARI(labels, labels_predict))


calc_and_output(sim_dis)
calc_and_output(sim_cor)
calc_and_output(sim_mu)
calc_and_output(sim_data)
