from Distance import SimDistance, SimCorrelation, SimMutual, Similarity
from ReadData import read_from_csv
import numpy as np
from Clustering import hca, hca_dendrogram, hca_labels
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from Metrics import ARI, accuracy
from sklearn.neighbors import KNeighborsClassifier

# read data
data = read_from_csv('data/yang_human_embryo.csv')
# read labels
labels = data.T[:90, 0]
labels = [i.partition('#')[0] for i in labels]

X = data.T[:90, 1:]
X = X.astype(np.float64)
print(X.shape)

sim_dis = SimDistance(X)
sim_cor = SimCorrelation(X)
sim_mu = SimMutual(X)
sim_data = Similarity(X, alpha=0.1, beta=0.9, gamma=0)


def calc_and_output(sim):
    # model = hca(sim)
    # labels_predict = hca_labels(model, 6)
    knn_model = KNeighborsClassifier(n_neighbors=6)
    knn_model.fit(sim, labels)
    labels_predict = knn_model.predict(sim)

    print('accuracy:', accuracy(labels, labels_predict))


calc_and_output(sim_dis)
calc_and_output(sim_cor)
calc_and_output(sim_mu)
calc_and_output(sim_data)
