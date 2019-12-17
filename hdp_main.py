# -*- coding: utf-8 -*-

"""
Perform the heterogeneous defect prediction to vertify the proposed method.
@author: Wenzhou Zhang
@date: 2019/3/8
"""

import scipy.io as sio
import numpy as np
import pandas as pd
from pre_proccessing import z_score_norm, log_trans
from VAE import VAE_HDP
import gc
import argparse
import multiprocessing

# dataset_rest =['poi', 'redaktor', 'ant',  'tomcat', 'xerces', 'skarbonka', 'velocity', 'camel',  'xalan', 'arc']

company = ['NASA', 'SOFTLAB', 'RELINK', 'AEEEM']
datasets = {'NASA': ['CM1', 'MW1', 'PC1', 'PC3', 'PC4'], 'SOFTLAB': ['AR1', 'AR3', 'AR4', 'AR5', 'AR6'],
            'RELINK': ['Apache', 'Safe', 'Zxing'], 'AEEEM': ['EQ', 'JDT', 'LC', 'ML', 'PDE']}


def vae_main(args):

    # the company of the target project
    for tar_cpy in ['AEEEM']:
        target_dataset = datasets[tar_cpy]
        # if tar_cpy == 'NASA':
        #     continue
        # the target projects
        for curr_tar in ['JDT']:
            # if curr_tar != 'Safe':
            #     continue
            # if curr_tar == 'Safe':
            #     first = False
            args.target_name = curr_tar
            target_data = sio.loadmat('dataset/' + curr_tar)
            for sou_cpy in ['NASA']:
                if tar_cpy == sou_cpy:
                    continue
                else:
                    source_dataset = datasets[sou_cpy]
                    # the source projects
                    for curr_sou in ['PC4']:
                        # if curr_sou != 'ML':
                        #     continue
                        print(curr_sou, '==>', curr_tar)
                        args.source_name = curr_sou
                        source_data = sio.loadmat('dataset/'+curr_sou)
                        model = VAE_HDP(args)
                        model.fit(source_data, target_data)
                        # model.fit(source_data, target_data)
                        del source_data, model
            del target_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # working_dir = osp.dirname(osp.abspath(__file__))
    working_dir = ''
    parser.add_argument('--logs_path', type=str, metavar='PATH',
                        default='logs')
    parser.add_argument('--model_path', type=str, metavar='PATH',
                        default='best_models')
    parser.add_argument('--score_path', type=str, metavar='PATH',
                        default='score/vae')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--source_name', type=str, default=None)
    parser.add_argument('--target_name', type=str, default=None)

    # Training args
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-1)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=50)

    # Net args
    parser.add_argument('--lambda_kl', type=float, default=1)
    parser.add_argument('--lambda_mmd', type=float, default=1)
    parser.add_argument('--lambda_ent', type=float, default=1e-2)
    parser.add_argument('--n_input', type=int, default=None)
    parser.add_argument('--n_hidden', type=int, default=None)
    parser.add_argument('--n_latent', type=int, default=20)
    parser.add_argument('--n_out', type=int, default=5)
    parser.add_argument('--activation', type=str, default='relu')

    args = parser.parse_args()
    vae_main(args)
