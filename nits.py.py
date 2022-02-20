#Author__Farhad_Mirkarimi -*- coding: utf-8 -*-
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
def mlp(dim, hidden_dim, output_dim, layers, activation):
    """Create a mlp from the configurations."""
    activation = {
        'relu': nn.ReLU,
        'lrelu': nn.LeakyReLU
    }[activation]

    seq = [nn.Linear(dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)

class PeakConstraint(nn.Module):
  """Implements an activation for peak constraint """
  def __init__(self, peak, **extra_kwargs):
    super(PeakConstraint, self).__init__()
    self.peak_activation = nn.Threshold(-peak, -peak)
  def forward(self, x):
    x = self.peak_activation(x)
    neg1 = torch.tensor(-1.0)
    x = neg1 * x
    x = self.peak_activation(x)
    x = neg1 * x
    return x


class NIT(nn.Module):
  """NIT """
  def __init__(self, dim, hidden_dim, layers, activation, avg_P,chan_type, peak=None,positive=None, **extra_kwargs):
    super(NIT, self).__init__()
    self._f = mlp(dim, hidden_dim, dim, layers, activation)
    self.avg_P = torch.tensor(avg_P)  # average power constraint
    self.peak = peak  # peak constraint  
    self.positive=positive
    self.chan_type=chan_type
    if self.peak is not None:
      
      self.peak_activation = PeakConstraint(peak)

  def forward(self, x):
   
    if self.chan_type=='conts_awgn':
        unnorm_tx = self._f(x)

        norm_tx = unnorm_tx/torch.sqrt(torch.mean(torch.pow(unnorm_tx,2.0)))*torch.sqrt(self.avg_P)
        

        if self.peak is not None:
          norm_tx = self.peak_activation(norm_tx)
        if self.positive is not None:
          norm_tx=F.softplus(norm_tx)

        return norm_tx
    
    
    
      if self.positive is not None:
        norm_tx=(torch.cosh(norm_tx))-1.0
      #norm_tx=self.ps(norm_tx)
      if self.peak is not None:
        norm_tx=self.peak_activation(norm_tx)
  
      
    
    
      
        return norm_tx
      

class _Channel(nn.Module):
  """AWGN Channel """
  def __init__(self,type1):
    super(_Channel, self).__init__()
    self.stdev = torch.tensor(1.0,dtype=torch.float)
    self.type1=type1
    ##
  def forward(self, x):
   
    if self.type1=='conts_awgn':
       noise = torch.randn_like(x) * self.stdev
       return x + noise
   
