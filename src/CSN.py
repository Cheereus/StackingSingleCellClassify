import joblib
import numpy as np
import math

# Gene Expression Matrix
GEM = joblib.load('datasets/Chu_cell_type.pkl').T
print(GEM.shape)
# Genes
genes = joblib.load('datasets/Chu_cell_type_genes.pkl')
print(len(genes))
# Cell Type Labels
labels = joblib.load('datasets/Chu_cell_type_labels.pkl')
print(len(labels))

# Preprocession of GEM
GEM_P = []
genes_P = []
idx = 0
for gene in GEM:
    gene = np.array(gene).astype(np.float64)
    if len(gene[gene > 0]) >= 10:
        GEM_P.append(gene)
        genes_P.append(genes[idx])
    idx += 1

GEM_P = np.array(GEM_P)

# 内存不爆，但只能得到最终的 NDM
n, n_genes = GEM_P.T.shape
nxk = int(0.1 * n)
nyk = int(0.1 * n)
mu_xyk = 0
sigma_xyk = math.sqrt((nxk * nyk * (n - nxk) * (n - nyk)) / (n * n * n * n * (n - 1)))
significant_level = 0.01

NDM = np.zeros((n_genes, n))

for k in range(n):
    single_cell_CSN = np.zeros((n_genes, n_genes))
    for x in range(n_genes):
        for y in range(x + 1, n_genes):
            if x == y:
                continue
            gene_x_dis = [abs(_ - GEM_P[x][k]) for _ in GEM_P[x]]
            gene_y_dis = [abs(_ - GEM_P[y][k]) for _ in GEM_P[y]]
            gene_x_knn = np.argsort(gene_x_dis)[1:nxk + 1]
            gene_y_knn = np.argsort(gene_y_dis)[1:nyk + 1]
            nxyk = len([x for x in gene_x_knn if x in gene_y_knn])
            ro_xyk = nxyk / n - (nxk / n) * (nyk / n)
            roxyk_hat = (ro_xyk - mu_xyk) / sigma_xyk

            edge_xyk = 0
            if roxyk_hat > significant_level:
                edge_xyk = 1

            single_cell_CSN[x][y] = edge_xyk

    for x in range(n_genes):
        NDM[x][k] = sum(single_cell_CSN[x])
    print(k)

print(NDM)
joblib.dump(NDM, 'Chu_NDM.pkl')
