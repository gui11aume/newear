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
#MSZ = 1698
MSZ = 1203

class XData(torch.utils.data.Dataset):

   def __init__(self, path, device='cpu'):

      super(XData, self).__init__()

      self.data = torch.zeros((MSZ,MSZ), device=device)
      with gzip.open(path) as f:
         for line in f:
            (a,b,c) = (int(_) for _ in line.split())
            self.data[a/WSZ-1,b/WSZ-1] = c


class Model(nn.Module):

   def __init__(self):

      super(Model, self).__init__()

      self.A = nn.Parameter(torch.ones(1))
      self.B = nn.Parameter(torch.ones(1))
      self.p = nn.Parameter(torch.ones(MSZ))
      #self.s = nn.Parameter(torch.ones(MSZ))
      self.l = nn.Parameter(torch.ones(1))
      self.t = nn.Parameter(torch.ones(1))
      # Mask half of the matrix, plus the diagonal.
      self.mask = torch.ones((MSZ,MSZ)).triu(1)
      if os.path.isfile('dmat.tch'):
         with open('dmat.tch') as f:
            self.dmat = torch.load(f)
      else:
         self.dmat = torch.zeros((MSZ,MSZ))
         for i in range(MSZ):
            for j in range(i, MSZ):
               self.dmat[i,j] = j-i+1
         with open('dmat.tch', 'w') as f:
            torch.save(self.dmat, f)


   def optimize(self, path):
      # Move to proper device.
      self.dmat = self.dmat.to(device=self.A.device)
      self.mask = self.mask.to(device=self.A.device)
      HiC = XData(path, device=self.A.device)
      # Weed out.
      out = torch.sum(HiC.data, 1) < 1
      optim = torch.optim.Adam(self.parameters(), lr=.01)
      sched = torch.optim.lr_scheduler.MultiStepLR(optim,
            milestones=[3000])
      # Take log distance.
      self.dmat[self.dmat == 0] = 1.0
      self.dmat = torch.log(self.dmat)
      s = torch.sqrt(torch.sum(HiC.data, 1))
      SS = torch.ger(s,s)
      for step in range(3200):
         # Take the sigmoid to constraint 'P' within (0,1).
         P = torch.sigmoid(self.p)
         x = P*self.A - (1-P)*self.A
         AB = torch.ger(x,x)
         # Expected counts.
         mu = SS * torch.exp(AB - self.dmat*self.l)
         # Reparametrize for the negative binoimal.
         nbp = mu / (mu + self.t)
         log_p = NegativeBinomial(self.t, nbp).log_prob(HiC.data)
         ll = -torch.sum(log_p * self.mask)
         optim.zero_grad()
         ll.backward()
         optim.step()
         sched.step()
         sys.stderr.write('%d %f\n' % (step, float(ll)))
         
      print '# A', float(self.A)
      print '# B', float(self.B)
      print '# l', float(self.l)
      print '# t', float(self.t)
      for i in range(MSZ):
         print i, float(torch.sigmoid(self.p[i])), float(out[i])

M = Model().cuda() if torch.cuda.is_available() else Model()
M.optimize(sys.argv[1])
