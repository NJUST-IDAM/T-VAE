# -*- coding: utf-8 -*-

"""
Preproccessing the defect data by performing the z-score normalization or log transformation
@author: Wenzhou Zhang
@date: 2019/3/19
"""

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import scipy.io as sio
import os


def z_score_norm(data):
    """
    :return:
    """
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data


def log_trans(data):
    """
    :param data:
    :return:
    """
    transformer = FunctionTransformer(np.log1p)
    data = transformer.transform(data)
    return data


def data_extra():
    path = 'dataset/dataset/'
    fn = os.listdir(path)
    data2 = sio.loadmat('dataset/Apache.mat')
    for f in fn:
        data = sio.loadmat(path+f)
        Xy = data['tar_set']
        y = Xy[-1, :].astype(np.int32)
        X = Xy[:-1, :].T
        sio.savemat(path+''+f, {'X': X, 'y': y})
        print('hh')


def data_verify():
    data1 = sio.loadmat('dataset/PDE.mat')
    data2 = sio.loadmat('dataset/dataset/PDE.mat')
    print(np.where(data1['X']!=data2['X']))


