#!/usr/bin/env python
# coding: utf-8

# In[1]:


def all_params(dim,layers_critic,embed_dim,hidden_dim_critic,activation_F1,lr_critic,dim_NIT,layers_NIT,hidden_dim_NIT,t_x_power,lr_NIT,channel_type,peak_amp,positive):
    critic_params = {
    'dim': dim,
    'layers': layers_critic,
    'embed_dim': embed_dim,
    'hidden_dim': hidden_dim_critic,
    'activation': activation_F1,
    'ref_batch_factor': 10,
    'learning_rate': lr_critic}


    nit_params = {'dim': dim_NIT,
              'layers': layers_NIT,
              'hidden_dim': hidden_dim_NIT,
              'activation': 'relu',
              'tx_power': t_x_power,
              'learning_rate': lr_NIT,
              'channel':channel_type,
              'peak_amp': peak_amp,
              'positive':positive}
    return nit_params,critic_params

