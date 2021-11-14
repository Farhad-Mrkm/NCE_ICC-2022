# author__Farhad_Mirkarimi-*- coding: utf-8 -*-
import os
import h5py
import glob, os
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange
from numpy.random import default_rng
import torch.nn.functional as F
import argparse
import gc
gc.collect()
print(np.version.version)
from all_params import all_params
from joint_training import joint_training
################ parsing input args###################

parser = argparse.ArgumentParser(description='provide arguments for neural capacity estimation')
#parser.add_argument('--SNR',        type=int,          default=[10],     help='Signal to noise(unit)')
parser.add_argument('--SNR',nargs='+',type=int)
parser.add_argument('--init_epoch',              type=int,       default=100,     help='First round epoch')
parser.add_argument('--max_epoch',              type=int,       default=3000,     help='joint training epoch')
parser.add_argument('--seed_size',              type=int,       default=2,     help='seed size for discrete inputs')
parser.add_argument('--batch_size',     type=int,       default=256,     help='batch size')
parser.add_argument('--hidden_dim_critic',      type=int,       default=256,     help='hidden dim for mi_est net')
parser.add_argument('--hidden_dim_nit',     type=int,       default=256,     help='hidden_dim for nit net')
parser.add_argument('--dim',     type=int,       default=1,     help='dimension for mi_est net')
parser.add_argument('--dim_nit',      type=int,     default=1,     help='dimension for NIT net')
parser.add_argument('--layer_mi',              type=int,     default=4,     help='layer number for mi_est net')
parser.add_argument('--layer_nit',              type=int,     default=4,     help='layer number for nit_net')
parser.add_argument('--lr_rate_nit',     type=float,     default=.0001,     help='training lr')
parser.add_argument('--lr_rate_mi_est',    type=float,     default=.00001,     help='training lr')
parser.add_argument('--type_channel',            type=str,       default='conts_awgn',     help='channel name')
parser.add_argument('--estimator',               type=str,        default='mine',      help='estimator type')
parser.add_argument('--activation',              type=str,     default='relu', help='activation function')
parser.add_argument('--peak',                    type=float,   default=None, help='peak_amplitude constraint')
parser.add_argument('--positive',                type=float,   default=None, help='positivity of input')
#parser.add_argument('--verbose',        dest='verbose', action='store_true')
#parser.set_defaults(verbose=False)

args = parser.parse_args()

######################################################3



nit_params,critic_params=all_params(dim=args.dim,layers_critic=args.layer_mi,embed_dim=32,hidden_dim_critic=256,activation_F1='relu',lr_critic=.0001,dim_NIT=args.dim_nit,layers_NIT=args.layer_nit,hidden_dim_NIT=256,t_x_power=1,lr_NIT=.0001,channel_type=args.type_channel,peak_amp=args.peak,positive=args.positive)
batch_x0,cap= joint_training(typeinp=args.type_channel,nit_params=nit_params,critic_params=critic_params,SNR=args.SNR,estimator=args.estimator,init_epoch=args.init_epoch,max_epoch=args.max_epoch,itr_every_nit=2,itr_every_mi=5,batch_size=args.batch_size,seed_size=args.seed_size)
