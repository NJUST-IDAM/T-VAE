# -*- coding: utf-8 -*-

"""
@author: Wenzhou Zhang
@date: 2019/3/21
"""

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon


company = ['NASA', 'SOFTLAB', 'RELINK', 'AEEEM']
datasets = {'NASA': ['CM1', 'MW1', 'PC1', 'PC3', 'PC4'], 'SOFTLAB': ['AR1', 'AR3', 'AR4', 'AR5', 'AR6'],
            'RELINK': ['Apache', 'Safe', 'Zxing'], 'AEEEM': ['EQ', 'JDT', 'LC', 'ML', 'PDE']}


def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q


def calculate_pvalue(X, y):
    """
    :param X: (n, m)
    :param y: (n, )
    :return:
    """
    res = list()
    for clo in X:
        res.append(wilcoxon(clo, y)[1])
    return p_adjust_bh(res)


def cliff_delta_value(X, Y):
    """
    Calculates Cliff's Delta function, a non-parametric effect magnitude
    test. See: http://revistas.javeriana.edu.co/index.php/revPsycho/article/viewFile/643/1092
    for implementation details.
    :param X:
    :param Y:
    :return:
    """

    # calculate length of vetors.
    lx = len(X)
    ly = len(Y)

    # comparison matrix. First dimension represnt elements in X, the second elements in Y
    # Values calculated as follows:
    # mat(i,j) = 1 if X(i) > Y(j), zero if they are equal, and -1 if X(i) < Y(j)
    mat = np.zeros((lx, ly))

    # perform all the comparisons.
    for i in range(lx):
        for j in range(ly):
            if X[i] > Y[j]:
                mat[i, j] = 1
            elif Y[j] > X[i]:
                mat[i, j] = -1

    # calculate delta.
    delta = {}
    value = np.abs(np.sum(mat) / (lx * ly))
    if value >= 0.474:
        delta['magnitude'] = 'Large'
    elif value >= 0.33:
        delta['magnitude'] = 'Medium'
    elif value >= 0.147:
        delta['magnitude'] = 'Small'
    else:
        delta['magnitude'] = 'Negligible'

    delta['estimate'] = value

    return delta


def cal_status():
    result = pd.read_csv('result.csv', index_col=0)
    base = result.pop('T-VAE')
    p_values = list()
    for clo in result.columns:
        print(clo)
        clo_value = result[clo]
        p_val = wilcoxon(clo_value, base.values)[1]
        print('p value:', p_val)
        print('cliff delta:', cliff_delta_value(clo_value.values, base.values))
        p_values.append(p_val)
    print(p_adjust_bh(p_values))


# if __name__ == '__main__':
#     cal_status()
