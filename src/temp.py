import joblib
import datetime
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

from Distance import SimDistance, SimCorrelation, SimMutual, Similarity, RelevanceMatrix
from DimensionReduction import t_SNE, get_pca
import numpy as np
from Clustering import hca, hca_dendrogram, hca_labels, k_means
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from Metrics import ARI, accuracy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering

dataset = "Xin_human_islets"
pic_title = dataset + " with pca 5"
X = joblib.load('datasets/' + dataset + '.pkl')
labels = joblib.load('datasets/' + dataset + '_labels.pkl')
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
    model = AgglomerativeClustering(n_clusters=8, affinity='euclidean')
    labels_predict = model.fit(sim).labels_
    # labels_predict = knn_model.predict(np.max(sim) - sim)
    print('ARI:', ARI(labels, labels_predict))
    return labels_predict, ARI(labels, labels_predict)


# sim_data = Similarity(dim_data, alpha=0.3, beta=0.6)
# pred, ari = calc_and_output(sim_data)
# rel = RelevanceMatrix(pred)
# sim_next = 0.6 * sim_data + 0.3 * rel
# l0 = len(sim_next[sim_next != 0])
# while l0 != 0:
#     print(l0)
#     sim_data = Similarity(sim_next, alpha=0.3, beta=0.6)
#     pred, ari = calc_and_output(sim_data)
#     rel = RelevanceMatrix(pred)
#     sim_next = 0.6 * sim_data + 0.3 * rel
#     l0 = len(sim_next[sim_next > 0])

# calc_and_output(sim_dis)
# calc_and_output(sim_cor)
# calc_and_output(sim_mu)

x = []
y = []
z = []
step = 0.1
n_steps = int(1 / step) + 1
m_z = np.zeros((n_steps, n_steps))

range_index = np.arange(0, 1.1, 0.1)

range_index = [round(i, 2) for i in range_index]

for i in range(n_steps):

    for j in range(n_steps - i):

        alpha = step * i
        beta = step * j
        sim_data = Similarity(dim_data, alpha=alpha, beta=beta)
        _, ari = calc_and_output(sim_data)
        x.append(alpha)
        y.append(beta)
        z.append(ari)
        m_z[j, i] = round(ari, 3)

print(m_z)
# heatmap
fig = plt.figure()
ax = fig.add_subplot(111)

# 定义横纵坐标的刻度
ax.set_yticks(range(len(range_index)))
ax.set_yticklabels(list(reversed(range_index)))
ax.set_xticks(range(len(range_index)))
ax.set_xticklabels(range_index)
im = ax.imshow(np.flipud(m_z), cmap=plt.cm.hot_r)
# 增加右侧的颜色刻度条
plt.colorbar(im)
# 增加标题
plt.title(pic_title)
# show
ax.set_ylabel('beta')
ax.set_xlabel('alpha')
plt.show()

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.plot_trisurf(x, y, z, cmap='rainbow')
# ax.set_zlabel('ARI')  # 坐标轴
# ax.set_ylabel('beta')
# ax.set_xlabel('alpha')
#
# plt.show()
