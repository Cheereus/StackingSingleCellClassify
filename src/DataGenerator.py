# data loader
# need data and labels
# data : samples * features
# return batched data and labels
import numpy as np
import joblib


# drop_last=True discard the last batch if size < batch_size
def Batch_2(data, labels, batch_size=10, drop_last=False, shuffle=True):
    data = np.array(data)
    labels = np.array(labels)
    n_samples, n_features = data.shape

    if shuffle:
        sf_idx = list(range(n_samples))
        np.random.shuffle(sf_idx)
        data = data[sf_idx]
        labels = labels[sf_idx]

    n = n_samples * (n_samples - 1) / 2
    batched_data = []
    batched_labels = []
    idx = 0
    batch_idx = 0

    for i in range(n_samples):
        for j in range(i + 1, n_samples):

            batched_sample = np.vstack((data[i], data[j]))
            batched_label = 1 if labels[i] == labels[j] else 0
            batched_data.append(batched_sample)
            batched_labels.append(batched_label)

            if len(batched_data) == batch_size or (batch_idx == (n // batch_size) and len(batched_data) == int(n % batch_size) and not drop_last):
                batch_idx += 1
                yield np.array(batched_data), np.array(batched_labels)
                batched_data = []
                batched_labels = []

            idx += 1


PBMC = joblib.load('datasets/Biase_mouse.pkl')
PBMC_labels = joblib.load('datasets/Biase_mouse_labels.pkl')
print(PBMC.shape)
dataLoader = Batch_2(PBMC, PBMC_labels)
for k in range(999999):
    next(dataLoader)


