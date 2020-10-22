import joblib
import datetime
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Distance import SimDistance, SimCorrelation, SimMutual, Similarity
from DimensionReduction import t_SNE, get_pca
import numpy as np
from Clustering import hca, hca_dendrogram, hca_labels, k_means
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from Metrics import ARI, accuracy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering

X = joblib.load('datasets/Li_islet.pkl')
labels = joblib.load('datasets/Li_islet_labels.pkl')
print(X.shape)
print(datetime.datetime.now())

dim_data, ratio, result = get_pca(X, c=5, with_normalize=True)
# print(ratio)
# dim_data = t_SNE(X, dim=2, perp=25, with_normalize=True)
# dim_data = X

# sim_dis = SimDistance(dim_data)
# sim_cor = SimCorrelation(dim_data)
# sim_mu = SimMutual(dim_data)


def calc_and_output(sim):
    # labels_predict = k_means(sim, 6)
    model = AgglomerativeClustering(n_clusters=6, affinity='euclidean')
    labels_predict = model.fit(sim).labels_
    # labels_predict = knn_model.predict(np.max(sim) - sim)
    print('ARI:', ARI(labels, labels_predict))
    return ARI(labels, labels_predict)


# calc_and_output(sim_dis)
# calc_and_output(sim_cor)
# calc_and_output(sim_mu)

x = []
y = []
z = []
t = 0

for alpha in np.arange(0, 1.01, 0.01):

    for beta in np.arange(0, 1.01 - alpha, 0.01):

        sim_data = Similarity(dim_data, alpha=alpha, beta=beta)
        ari = calc_and_output(sim_data)
        x.append(alpha)
        y.append(beta)
        z.append(ari)
        t += 1
        print(t)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_trisurf(x, y, z, cmap='rainbow')
ax.set_zlabel('ARI')  # 坐标轴
ax.set_ylabel('beta')
ax.set_xlabel('alpha')

plt.show()
