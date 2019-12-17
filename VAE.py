# -*- coding: utf-8 -*-
# @date: 2019/11/25
# @author: Wenzhou Zhang


"""
Using variational auto encoder to learn the coincident latent feature space.
"""

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from pre_proccessing import log_trans

from tensorflow import keras
import tensorflow as tf

import numpy as np
import pandas as pd
import tensorflow_probability as tfp
import multiprocessing
import logging
import time
import os


class VAE_HDP(object):

    def __init__(self, net_args=None):
        """
        :param net_args: the parameters of learned network
        """
        self.net_args = net_args
        self.Sx = None
        self.Sy = None
        self.Tx = None
        self.Ty = None

        self.workers = net_args.workers

        self.kl_range = [1e-2, 1e-1, 1e0, 1e+1, 1e+2]
        self.mmd_range = [1e-2, 1e-1, 1e0, 1e+1, 1e+2]
        self.ent_range = [1e-2, 1e-1, 1e0, 1e+1, 1e+2]

    def fit(self, source, target):
        self._init_data(source, target)

        # parameter package
        pool = multiprocessing.Pool(self.net_args.workers)
        args_li = []
        for kl in self.kl_range:
            for mmd in self.mmd_range:
                for ent in self.ent_range:
                    if not os.path.isfile(os.path.join(self.net_args.score_path, self.net_args.target_name, self.net_args.source_name, 'kl_{}-mmd_{}-ent_{}'.format(kl, mmd, ent))):
                        args_li.append({'kl': kl, 'mmd': mmd, 'ent': ent})
        pool.map(self._train, args_li)
        pool.close()
        pool.join()
        print('{}=>{} finished!'.format(self.net_args.source_name, self.net_args.target_name))

    def _init_data(self, s, t):
        """
        Initialize the data and parameters.
        """
        sx, sy = s['X'], np.ravel(s['y'])
        tx, ty = t['X'], np.ravel(t['y'])
        # 互补扩充
        # self.Sx = np.concatenate((sx, np.zeros(shape=(np.shape(sx)[0], np.shape(tx)[1]))), axis=1)
        # self.Tx = np.concatenate((np.zeros(shape=(np.shape(tx)[0], np.shape(sx)[1])), tx), axis=1)
        # n_input = self.Sx.shape[1] + self.Tx.shape[1]
        # 横向补零
        if np.shape(tx)[1] > np.shape(sx)[1]:
            sx = np.concatenate((sx, np.zeros(shape=(len(sx), np.shape(tx)[1] - np.shape(sx)[1]))), axis=1)
        else:
            tx = np.concatenate((tx, np.zeros(shape=(len(tx), np.shape(sx)[1] - np.shape(tx)[1]))), axis=1)

        self.Sx = log_trans(sx)
        self.Tx = log_trans(tx)
        self.Sy = sy
        self.Ty = ty
        self.net_args.n_input = self.Sx.shape[1]
        self.net_args.n_hidden = self.Sx.shape[1] // 3 * 2

    def _train(self, para):
        self.net_args.lambda_kl = para['kl']
        self.net_args.lambda_mmd = para['mmd']
        self.net_args.lambda_ent = para['ent']
        net = VaeNet(self.net_args, pd.Series())
        net.fit(self.Sx, self.Tx, self.Sy, self.Ty)
        print('kl:{} \tmmd:{} \tent:{} finished'.format(para['kl'], para['mmd'], para['ent']))

    def auc_score(self, target_Y):
        """
        Get the auc score of our model
        :param target_Y: The real label of the target project
        :return: auc score
        """
        return roc_auc_score(target_Y, self.Ty_pre)

    def f1_score(self, target_Y):
        from sklearn.metrics import accuracy_score
        return f1_score(target_Y, self.By_pre)


class VaeNet(keras.Model):
    """
    Variational auto encoder
    """

    def __init__(self, args=None, notebook=None):
        super(VaeNet, self).__init__()
        tf.keras.backend.set_floatx('float64')
        tf.random.set_seed(args.seed)
        self.n_input = args.n_input
        self.n_hidden = args.n_hidden
        self.n_latent = args.n_latent
        self.n_out = args.n_out
        self.activation = args.activation
        self.weight_decay = args.weight_decay
        self.optimizer = keras.optimizers.Adam(args.lr)
        self.epoch = args.epochs
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq

        self.logs_path = os.path.join(args.logs_path, args.target_name, args.source_name)
        self.model_path = os.path.join(args.model_path, args.target_name, args.source_name)
        self.score_path = os.path.join(args.score_path, args.target_name, args.source_name)

        self.inference_network = keras.Sequential([keras.layers.InputLayer(input_shape=(self.n_input,)),
                                                   keras.layers.Dense(self.n_hidden, activation=self.activation),
                                                   keras.layers.Dense(self.n_hidden, activation=self.activation),
                                                   keras.layers.Dense(self.n_latent * 2, activation=None)])
        self.generative_network = keras.Sequential([keras.layers.InputLayer(input_shape=(self.n_latent,)),
                                                    keras.layers.Dense(self.n_hidden, activation=self.activation),
                                                    keras.layers.Dense(self.n_hidden, activation=self.activation),
                                                    keras.layers.Dense(self.n_input, activation=None)])
        self.classification_network = keras.Sequential([keras.layers.InputLayer(input_shape=(self.n_latent,)),
                                                        keras.layers.Dense(self.n_out, activation='sigmoid')])
        self.lambda_mmd = args.lambda_mmd
        self.lambda_ent = args.lambda_ent
        self.lambda_kl = args.lambda_kl
        self.formatter = 'kl_{}-mmd_{}-ent_{}'.format(self.lambda_kl, self.lambda_mmd, self.lambda_ent)
        self.notebook = notebook
        self._check_path()
        self.logger = self._logger()

    def loss(self, sx, tx, sy, ty):

        inputs = tf.concat((sx, tx), axis=0)
        lat_feature = self.inference_network(inputs)

        mean = lat_feature[:, :self.n_latent]
        var = tf.math.softplus(lat_feature[:, self.n_latent:])

        z_distribution = tfp.distributions.Normal(loc=mean, scale=var)
        # Standard normal distributions with mean=0 and stddev=1
        std_distribution = tfp.distributions.Normal(loc=np.zeros(self.n_latent), scale=np.ones(self.n_latent))

        with tf.name_scope('samples_generation'):
            z_samples = z_distribution.sample()  # generate samples from origin latent distribution

        outputs = self.generative_network(z_samples)

        label = self.classification_network(mean)

        with tf.name_scope('kl_divergence'):
            # ****** the kl divergence of the posterior density from standard normal distribution *******
            # the value is calculated according to the parameters
            kl = tf.math.reduce_mean(tfp.distributions.kl_divergence(z_distribution, std_distribution), 1)

        with tf.name_scope('mmd'):
            mmd = tf.linalg.trace(tf.matmul(tf.matmul(tf.transpose(mean), self._M(sy.shape[0], ty.shape[0])), mean))

        with tf.name_scope('cross_entropy'):
            entory = tf.reduce_mean(keras.losses.binary_crossentropy(np.repeat([sy], self.n_out, axis=0).T,
                                                                     label[:sy.shape[0]]))
            tar_entory = tf.reduce_mean(keras.losses.binary_crossentropy(np.repeat([ty], self.n_out, axis=0).T,
                                                                         label[sy.shape[0]:]))

        with tf.name_scope('mean_square_error'):
            rec = keras.losses.mean_squared_error(inputs, outputs)  # 靠谱

        elbo = tf.math.reduce_sum(rec + self.lambda_kl * kl)

        return elbo + self.lambda_mmd * mmd + self.lambda_ent * entory, tar_entory

    def fit(self, sx, tx, sy, ty):
        source_data = tf.data.Dataset.from_tensor_slices((sx, sy)).shuffle(self.batch_size).batch(self.batch_size)
        target_data = tf.data.Dataset.from_tensor_slices((tx, ty)).shuffle(self.batch_size).batch(self.batch_size)
        start_time = time.time()
        best_score = 0
        for epoch in range(self.epoch):
            cur_loss, cur_err = 0, 0
            for s_batch, t_batch in zip(source_data, target_data):
                with tf.GradientTape() as tape:
                    train_loss, tar_err = self.loss(s_batch[0], t_batch[0], s_batch[1], t_batch[1])
                cur_loss += train_loss
                cur_err += tar_err
                gradients = tape.gradient(train_loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            if epoch % self.print_freq == 0:
                self.logger.info(
                    'KL: {}, MMD: {}, ENT: {} --- epoch: {}, ELBO+ENT: {:5f}, TEST_ENT: {:5f}, time elapse {:5f}'
                        .format(self.lambda_kl, self.lambda_mmd, self.lambda_ent, epoch, cur_loss, cur_err,
                                time.time() - start_time))

                _code = self.inference_network(tx)
                auc = roc_auc_score(ty, tf.reduce_mean(self.classification_network(_code[:, :self.n_latent]), 1))
                self.notebook.loc[epoch] = auc
                if auc > best_score:
                    best_score = auc
                    self.save_weights(os.path.join(self.model_path, self.formatter))
                # if epoch % 200 == 0:
                #     self.optimizer.learning_rate = self.optimizer.learning_rate * self.weight_decay
        self.notebook.to_pickle(os.path.join(self.score_path, self.formatter))

    def _M(self, ns, nt):
        n = ns + nt
        M = np.full((n, n), (1 / (ns * nt)))
        M[:ns, :ns] = 1 / (ns ** 2)
        M[-ns:, -ns:] = 1 / (nt ** 2)
        return M

    def _logger(self):
        loggging_format = logging.Formatter('%(asctime)s - PID: %(process)d - %(message)s')
        logger = logging.getLogger(self.formatter)
        logger.setLevel(logging.INFO)

        sh = logging.StreamHandler()
        sh.setFormatter(loggging_format)
        sh.setLevel(logging.INFO)
        logger.addHandler(sh)

        fh = logging.FileHandler(os.path.join(self.logs_path, self.formatter + '.log'))
        fh.setFormatter(loggging_format)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        return logger

    def _check_path(self):
        if not os.path.isdir(self.logs_path):
            try:
                os.makedirs(self.logs_path)
            except:
                pass
        if not os.path.isdir(self.model_path):
            try:
                os.makedirs(self.model_path)
            except:
                pass
        if not os.path.isdir(self.score_path):
            try:
                os.makedirs(self.score_path)
            except:
                pass
