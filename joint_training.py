# author__Farhad_Mirkarimi!/usr/bin/env python
# coding: utf-8

# In[ ]:


def joint_training(typeinp,nit_params,critic_params,SNR,estimator='mine',init_epoch=100,max_epoch=1000,itr_every_nit=1,itr_every_mi=5,batch_size=256,seed_size=4):
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

  seed_rv_std = .1
  batch_size=256
  ###########
  if torch.cuda.is_available():  
      dev = "cuda:0" 
  else:  
      dev = "cpu"

  device = torch.device(dev)
  ##########
  
  from nits import NIT,_Channel,PeakConstraint
  from mi_est import MI_Est_Losses
  from mi_nets import ConcatCritic
  actual_cap=[]
  est_cap=[]
  SNRs=SNR
  clip=.2
  for snr_db in SNRs:
    nit_params['tx_power'] = np.power(10, snr_db / 10)
    capacity = .5 * np.log(1 + nit_params['tx_power'])
    actual_cap.append(capacity)

    #print('The TX power at SNR {} dB is {}'.format(snr_db, nit_params['tx_power']))
    #print('The capacity is {}'.format(capacity))
    print(SNRs)
    nit = NIT(nit_params['dim'], nit_params['hidden_dim'],
              nit_params['layers'], nit_params['activation'], 
              nit_params['tx_power'],chan_type=typeinp, peak=nit_params['peak_amp'],positive=nit_params['positive'])
    nit.to(device)

    opt_nit = torch.optim.Adam(nit.parameters(), lr=nit_params['learning_rate'])

    channel =_Channel(typeinp)
    channel.to(device)

    mi_est_loss = MI_Est_Losses(estimator, device)
    mi_neuralnet = ConcatCritic(critic_params['dim'], critic_params['hidden_dim'],
                                critic_params['layers'], critic_params['activation'])

    mi_neuralnet.to(device)

    opt_mi = torch.optim.Adam(mi_neuralnet.parameters(), lr=critic_params['learning_rate'])

    estimates = np.zeros(init_epoch)
    for e in tqdm(range(init_epoch)):
      
      if typeinp=='conts_awgn':
        seed_rv = np.random.randn(batch_size,1)*seed_rv_std
        batch_s = torch.tensor(seed_rv, dtype=torch.float).to(device) 
        batch_x = nit(batch_s)
        batch_y = channel(batch_x)
        
        t_xy=mi_neuralnet(batch_x,batch_y)  
        
         
        if estimator=='mine':
            mi_est = mi_est_loss.mi_est_loss(t_xy)
        elif estimator=='smile':
            mi_est=mi_est_loss.mi_est_loss(t_xy,clip=clip)
       
        loss = -mi_est
        loss.backward()
    #torch.nn.utils.clip_grad_norm_(mi_neuralnet.parameters(), 0.2)
        opt_mi.step()
        mi_est = mi_est.detach().cpu().numpy()
        estimates[e]= mi_est
     

    plt.plot(estimates)
    plt.show()

    estimates = np.zeros(max_epoch)
    for e in tqdm(range(max_epoch)):
      for i in range(itr_every_nit):
        opt_nit.zero_grad()
        opt_mi.zero_grad()
        ##
       ### seed_rv = np.random.randint(0,seed_size,(batch_size,2))*seed_rv_std
       ### batch_s = torch.tensor(seed_rv, dtype=torch.float).to(device) 
       ### batch_x = nit(batch_s)
       ### batch_y = awgn(batch_x)
        ###zer=torch.ones_like(batch_y)*.001
        ###zer1=zer.view(batch_size,1)
        ###batchy1=batch_y.view(batch_size,1)
        ##fd=torch.cat((batchy1,zer1),1)
        ###t_xy = mi_neuralnet(batch_x, fd)
        ###mi_est = mi_est_loss.mi_est_loss(t_xy)
        ###loss = -mi_est
        ###loss.backward()
      #torch.nn.utils.clip_grad_norm_(nit.parameters(), 0.2)
        
        if typeinp=='conts_awgn':
          seed_rv = np.random.randn(batch_size,1)*seed_rv_std
          batch_s = torch.tensor(seed_rv, dtype=torch.float).to(device) 
          batch_x = nit(batch_s)
          batch_y = channel(batch_x)
          t_xy = mi_neuralnet(batch_x, batch_y)
          if estimator=='mine':
            mi_est = mi_est_loss.mi_est_loss(t_xy)
          elif estimator=='smile':
            mi_est=mi_est_loss.mi_est_loss(t_xy,clip=clip)
        
          loss = -mi_est
          loss.backward()
          opt_nit.step()
        elif typeinp=='discrt_poisson':
          #clip=2
         # opt_nit.zero_grad()
         # opt_mi.zero_grad()
      ##
          seed_rv0 = np.random.randn(batch_size,1)*.001
      #seed_rv = np.reshape(np.random.poisson(1,batch_size),(batch_size,1))+
          seed_rv=np.random.randint(0,seed_size,size=(batch_size,1))*50
      #seed_rv=np.random.uniform(0,nit_params['peak_amp'],(batch_size,1))*50+seed_rv0
          batch_s = torch.tensor(seed_rv, dtype=torch.float).to(device) 
          batch_x = nit(batch_s)
          batch_y = channel(batch_x)
          t_xy = mi_neuralnet(batch_x, batch_y)##
          if estimator=='mine':
            mi_est = mi_est_loss.mi_est_loss(t_xy)
          elif estimator=='smile':
            mi_est=mi_est_loss.mi_est_loss(t_xy,clip=clip)
        
          loss = -mi_est
          loss.backward()
      #torch.nn.utils.clip_grad_norm_(nit.parameters(), 0.2)
          opt_nit.step()
          
          #mi_est = mi_est.detach().cpu().numpy()
          #estimates[e]= mi_est
          #mi_est = mi_est.detach().cpu().numpy()
        #opt_nit.step()
      
      for i in range(itr_every_mi):
        if typeinp=='discrt_rician':
          opt_nit.zero_grad()
          opt_mi.zero_grad()
          seed_rv = np.random.randint(0,seed_size,(batch_size,2))*seed_rv_std
          batch_s = torch.tensor(seed_rv, dtype=torch.float).to(device) 
          batch_x = nit(batch_s)
          batch_y = channel(batch_x)
          zer=torch.ones_like(batch_y)*.001
          zer1=zer.view(batch_size,1)
          batchy1=batch_y.view(batch_size,1)
          fd=torch.cat((batchy1,zer1),1)
          t_xy = mi_neuralnet(batch_x, fd)
          if estimator=='mine':
            mi_est = mi_est_loss.mi_est_loss(t_xy)
          elif estimator=='smile':
            mi_est=mi_est_loss.mi_est_loss(t_xy,clip=clip)
        
          loss = -mi_est
          loss.backward()
          ##
              
          #torch.nn.utils.clip_grad_norm_(mi_neuralnet.parameters(), 0.2)
          opt_mi.step()
          mi_est = mi_est.detach().cpu().numpy()

          estimates[e]= mi_est
        elif typeinp=='conts_awgn': 
          opt_nit.zero_grad()
          opt_mi.zero_grad()
          seed_rv = np.random.randn(batch_size,1)*seed_rv_std
          batch_s = torch.tensor(seed_rv, dtype=torch.float).to(device) 
          batch_x = nit(batch_s)
          batch_y = channel(batch_x)
          t_xy = mi_neuralnet(batch_x, batch_y)
          if estimator=='mine':
            mi_est = mi_est_loss.mi_est_loss(t_xy)
          elif estimator=='smile':
            mi_est=mi_est_loss.mi_est_loss(t_xy,clip=clip)
        
          loss = -mi_est
          loss.backward()
          opt_mi.step()
          mi_est = mi_est.detach().cpu().numpy()

          estimates[e]= mi_est
      ##
        
          

    plt.plot(estimates)
    plt.show()

    opt_nit.zero_grad()
    opt_mi.zero_grad() 

    

  ##########new function
   
    if typeinp=='conts_awgn':
      seed_rv = np.random.randn(10000,1)*seed_rv_std
      batch_s = torch.tensor(seed_rv, dtype=torch.float).to(device) 
      batch_x = nit(batch_s)
      batch_x = batch_x.detach().cpu().numpy()
      print("The variance of batch is {}".format(np.var(batch_x)))
      print("-Estimated Capacity {:.4f}".format(np.mean(estimates[-50:])))
      plt.hist(batch_x, 100)
      plt.show()
      #from google.colab import files
      plt.savefig("optim-inp-awgn.png")
      #files.download("hist.pdf")
      est_cap.append(np.mean(estimates[-50:]))
      f = open("summary.txt", "a")
      f.write("{:100s} channel capacity at snr,{:02d},{:2.5f}\n".format(typeinp,snr_db, np.mean(estimates[-50:])))
      f.close()
      #return batch_x,np.mean(estimates[-50:])

   
  return batch_x,np.mean(estimates[-50:])

