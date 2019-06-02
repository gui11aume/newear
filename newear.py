#!/usr/bin/env python

import gzip
import os
import sys

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data

from torch.distributions.negative_binomial import NegativeBinomial

WSZ = 100000

class XData(torch.utils.data.Dataset):

   def __init__(self, path, device='auto'):

      super(XData, self).__init__()

      # Use graphics card whenever possible.
      self.device = device if device != 'auto' else \
         'cuda' if torch.cuda.is_available() else 'cpu'

      # Store contacts in set 'S' to determine
      # the size 'sz' of the Hi-C contact matrix.
      S = set()
      sz = 0
      with gzip.open(path) as f:
         for line in f:
            (a,b,c) = (int(_) for _ in line.split())
            S.add((a,b,c))
            if a > sz: sz = a
            if b > sz: sz = b

      # Write Hi-C matrix as 'pytorch' tensor.
      self.sz = sz / WSZ
      self.data = torch.zeros((self.sz,self.sz), device=self.device)
      for (a,b,c) in S:
         self.data[a/WSZ-1,b/WSZ-1] = c


class Model(nn.Module):

   def __init__(self, HiC):
      super(Model, self).__init__()
      # Data.
      self.HiC = HiC
      self.sz = HiC.sz
      self.device = HiC.device
      # Parameters.
      self.p = nn.Parameter(torch.ones(self.sz, device=self.device))
      self.b = nn.Parameter(torch.ones(1, device=self.device))
      self.a = nn.Parameter(torch.ones(1, device=self.device))
      self.t = nn.Parameter(torch.ones(1, device=self.device))


   def optimize(self):
      # Mask the diagonal (dominant outlier) and half of
      # the matrix to not double-count the evidence.
      mask = torch.ones((self.sz,self.sz), device=self.device).triu(1)
      # Compute log-distances between loci.
      u = torch.ones((self.sz,self.sz), device=self.device).triu()
      dmat = torch.matrix_power(u, 2)
      dmat[dmat == 0] = 1.0
      dmat = torch.log(dmat)
      # Optimizer and scheduler.
      optim = torch.optim.Adam(self.parameters(), lr=.01)
      sched = torch.optim.lr_scheduler.MultiStepLR(optim,
            milestones=[3000])
      # Weights (mappability biases etc.)
      w = torch.sqrt(torch.sum(self.HiC.data, 1))
      W = torch.ger(w,w)
      for step in range(3200):
         # Take the sigmoid to constrain 'P' within (0,1).
         P = torch.sigmoid(self.p)
         x = P*self.b - (1-P)*self.b
         AB = torch.ger(x,x)
         # Expected counts.
         mu = W * torch.exp(AB - dmat*self.a)
         # Reparametrize for the negative binoimal.
         nbp = mu / (mu + self.t)
         log_p = NegativeBinomial(self.t, nbp).log_prob(self.HiC.data)
         # Multiply by mask to remove entries.
         ll = -torch.sum(log_p * mask)
         optim.zero_grad()
         ll.backward()
         optim.step()
         sched.step()
         sys.stderr.write('%d %f\n' % (step, float(ll)))
         
      print '# alpha', float(self.a)
      print '# beta', float(self.b)
      print '# theta', float(self.t)
      for i in range(self.sz):
         print i, float(torch.sigmoid(self.p[i]))

M = Model(XData(sys.argv[1]))
M.optimize()
