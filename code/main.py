import sys
import math
import torch
import ctypes
import datetime
import numpy as np
import argparse
import time
import random
import os

from model import MVTGC

FType = torch.FloatTensor
LType = torch.LongTensor

import warnings
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak on Windows with MKL")

def main_train(args):
    start = datetime.datetime.now()
    the_train = MVTGC.MVTGC(args)
    the_train.train()
    end = datetime.datetime.now()
    print('Training Complete with Time: %s' % str(end - start))


if __name__ == '__main__':

    data = 'arxivAI'
    k_dict = {'arxivAI': 5, 'school': 9, 'dblp': 10, 'brain': 10, 'patent': 6}
    RW_dict = {'arxivAI': 1, 'school': 0.8, 'dblp': 0.5, 'brain': 0.5, 'patent': 0.6}
    PE_dict = {'arxivAI': 0, 'school': 0.1, 'dblp': 0.3, 'brain': 0.5, 'patent': 0.1}
    View_RW_path = '../data/%s/View_RW.txt' % data
    View_PE_path = '../data/%s/View_PE.txt' % data
    View_A_path = '../data/%s/View_MP.txt' % data

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=data)
    parser.add_argument('--r_RW', type=str, default=RW_dict[data])
    parser.add_argument('--r_PE', type=str, default=PE_dict[data])
    parser.add_argument('--clusters', type=int, default=k_dict[data])
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--View_RW_path', type=str, default=View_RW_path)
    parser.add_argument('--View_PE_path', type=str, default=View_PE_path)
    parser.add_argument('--View_A_path', type=str, default=View_A_path)

    parser.add_argument('--neg_size', type=int, default=3)
    parser.add_argument('--hist_len', type=int, default=2)
    # [b:lr]=1024:0.01

    parser.add_argument('--save_step', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--directed', type=bool, default=False)
    args = parser.parse_args()

    print(args)
    main_train(args)