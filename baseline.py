import tensorflow as tf
import numpy as np
from tensorflow import keras
import scipy.io as sio
from sklearn.metrics import roc_auc_score
import random
import pandas as pd
import multiprocessing


company = ['NASA', 'SOFTLAB', 'RELINK', 'AEEEM']
datasets = {'NASA': ['CM1', 'MW1', 'PC1', 'PC3', 'PC4'], 'SOFTLAB': ['AR1', 'AR3', 'AR4', 'AR5', 'AR6'],
            'RELINK': ['Apache', 'Safe', 'Zxing'], 'AEEEM': ['EQ', 'JDT', 'LC', 'ML', 'PDE']}


class ClassifyNet():

    def __init__(self, n_latent, n_out):
        self.n_out = n_out
        self.model = keras.Sequential([keras.layers.InputLayer(input_shape=(n_latent,)),
                                       keras.layers.Dense(n_out, activation='sigmoid')])
        self.model.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(lr=1e-4), metrics=['accuracy'])

    def fit(self, x, y):
        index = np.array([False] * len(y))
        index[random.sample(range(len(y)), int(len(y)*0.8))] = True
        y = np.repeat([y], self.n_out, axis=0).T
        history = self.model.fit(x, y, epochs=1000, verbose=0)
        # print(history)

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        return roc_auc_score(y, np.mean(self.predict(x), 1))


if __name__ == '__main__':
    net_sc = pd.DataFrame()
    for i in range(50):
        for cpy in company:
            dataset = datasets[cpy]
            for curr_set in dataset:
                print(curr_set, i, 'start!')
                data = sio.loadmat('dataset/' + curr_set)
                x = data['X']
                y = np.ravel(data['y'])
                index = np.array([False]*len(y))
                index[random.sample(range(len(y)), int(len(y) * 0.5))] = True
                net = ClassifyNet(x.shape[1], 5)
                net.fit(x[index], y[index])
                net_sc.loc[i*2, curr_set] = net.score(x[~index], y[~index])
                del net
                net = ClassifyNet(x.shape[1], 5)
                net.fit(x[~index], y[~index])
                net_sc.loc[i*2+1, curr_set] = net.score(x[index], y[index])
                net_sc.to_csv('score/base_net.csv')
                del data, net
