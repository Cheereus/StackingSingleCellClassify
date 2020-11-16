import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from PIL import Image
from matplotlib import pyplot as plt
import joblib

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


h_dim = 20
batchsz = 20
lr = 1e-3

x_train = joblib.load('datasets/PBMC.pkl')
n_cells, n_genes = x_train.shape
y_train = np.array(joblib.load('datasets/PBMC_labels.pkl'))
# (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype(np.float32)

label_index = np.arange(x_train.shape[0])
print(label_index)
np.random.shuffle(label_index)
x_train = x_train[label_index]
y_train = y_train[label_index]
joblib.dump(y_train, 'ae_output/labels.pkl')

# we do not need label
train_db = tf.data.Dataset.from_tensor_slices(x_train)
train_db = train_db.batch(batchsz)

print(x_train.shape)

class AE(keras.Model):

    def __init__(self):
        super(AE, self).__init__()

        # Encoders
        self.encoder = Sequential([
            layers.Dense(512, activation=tf.nn.leaky_relu),
            layers.Dense(256, activation=tf.nn.leaky_relu),
            layers.Dense(128, activation=tf.nn.sigmoid),
            layers.Dropout(rate=0.2),
            layers.Dense(h_dim)
        ])

        # Decoders
        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.sigmoid),
            layers.Dense(256, activation=tf.nn.leaky_relu),
            layers.Dense(512, activation=tf.nn.leaky_relu),
            layers.Dense(n_genes)
        ])

    def call(self, inputs, training=None):
        # [b, 784] => [b, 10]
        h = self.encoder(inputs)
        # [b, 10] => [b, 784]
        x_hat = self.decoder(h)

        return x_hat, h


model = AE()
model.build(input_shape=(None, n_genes))
model.summary()

optimizer = tf.optimizers.Adam(lr=lr)

for epoch in range(100):

    for step, x in enumerate(train_db):

        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, [-1, n_genes])

        with tf.GradientTape() as tape:
            x_rec_logits, _ = model(x)

            rec_loss = tf.losses.MSE(x, x_rec_logits)
            rec_loss = tf.reduce_mean(rec_loss)

        grads = tape.gradient(rec_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 100 == 0:
            print(epoch, step, float(rec_loss))

        # evaluation
        # x = next(iter(test_db))
        # logits = model(tf.reshape(x, [-1, 784]))
        # x_hat = tf.sigmoid(logits)

    if epoch % 10 == 0 or epoch == 99:

        print('Saving epoch', epoch)
        dim_data = None
        for step, x in enumerate(train_db):
            _, x_reduced = model(x)
            if step == 0:
                dim_data = x_reduced
            else:
                dim_data = np.vstack((dim_data, x_reduced))

        print(epoch, dim_data.shape)
        joblib.dump(dim_data, 'ae_output/ae_dim_data_' + str(epoch) + '.pkl')
