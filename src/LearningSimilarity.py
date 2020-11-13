from abc import ABC

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import Sequential, layers
import joblib
from DataGenerator import Batch_2


data = joblib.load('datasets/Chu_cell_type.pkl')
labels = joblib.load('datasets/Chu_cell_type_labels.pkl')
print(data.shape)
cells, genes = data.shape

batch_sz = 10
lr = 1e-3


class AE(keras.Model, ABC):

    def __init__(self):
        super(AE, self).__init__()

        # Encoders
        self.encoder = Sequential([
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(128, activation=tf.nn.sigmoid),
            layers.Dropout(rate=0.2),
            layers.Dense(1)
        ])

        # Decoders
        self.decoder = Sequential([
            layers.Dense(128, activation=tf.nn.sigmoid),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(genes*2)
        ])

    def call(self, inputs, training=None):
        # [b, 784] => [b, 10]
        h = self.encoder(inputs)
        # [b, 10] => [b, 784]
        x_hat = self.decoder(h)

        return x_hat, h


model = AE()
model.build(input_shape=(None, genes*2))
model.summary()

optimizer = tf.optimizers.Adam(lr=lr)


for epoch in range(100):

    dataLoader = Batch_2(data, labels)
    for step, batch_data in enumerate(dataLoader):

        x = batch_data[0]
        x = tf.convert_to_tensor(x, tf.float32)
        x = tf.reshape(x, [-1, genes*2])

        with tf.GradientTape() as tape:
            x_rec_, _ = model(x)

            rec_loss = tf.losses.MSE(x, x_rec_)
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
        dataLoader2 = Batch_2(data, labels)
        for step, batch_data in enumerate(dataLoader2):

            x = batch_data[0]
            x = tf.convert_to_tensor(x, tf.float32)
            x = tf.reshape(x, [-1, genes*2])
            _, x_reduced = model(x)
            if step == 0:
                dim_data = x_reduced
            else:
                dim_data = np.vstack((dim_data, x_reduced))

        print(epoch, dim_data.shape)
        joblib.dump(dim_data, 'ae_output/sim_' + str(epoch) + '.pkl')
