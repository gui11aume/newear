#!/usr/bin/env python

import gzip
import os
import sys

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data

from torch.nn.functional import softplus
from torch.distributions.negative_binomial import NegativeBinomial

WSZ = 100000

class HiCData(torch.utils.data.Dataset):

   def __init__(self, path, sz=0, device='auto'):
      super(HiCData, self).__init__()
      # Use graphics card whenever possible.
      self.device = device if device != 'auto' else \
         'cuda' if torch.cuda.is_available() else 'cpu'
      if sz > 0:
         # The final size 'sz' is specified.
         self.sz = sz
         self.data = torch.zeros((self.sz,self.sz), device=self.device)
         self.add(path)
      else:
         # Store contacts in set 'S' to determine
         # the size 'sz' of the Hi-C contact matrix.
         S = set()
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
            A = a/WSZ-1
            B = b/WSZ-1
            self.data[A,B] = c
            self.data[B,A] = c

   def add(self, path):
      (maxA, maxB) = self.data.shape
      with gzip.open(path) as f:
         for line in f:
            (a,b,c) = (int(_) for _ in line.split())
            A = a/WSZ-1
            B = b/WSZ-1
            if A < maxA and B < maxB:
               self.data[A,B] += c
               self.data[B,A] += c

   def show_data(self, fout):
      for i in range(self.sz):
         row = '\t'.join(['%d' % int(x) for x in self.data[i,:]])
         fout.write(row + '\n')


class Model(nn.Module):

   def __init__(self, HiC):
      super(Model, self).__init__()
      # Data.
      self.HiC = HiC
      self.sz = HiC.sz
      self.device = HiC.device
      # Parameters.
      self.p = nn.Parameter(torch.ones(self.sz, device=self.device))
      self.a = nn.Parameter(torch.ones(1, device=self.device))
      self.b = nn.Parameter(torch.ones(1, device=self.device))
      self.g = nn.Parameter(torch.ones(1, device=self.device))
      self.t = nn.Parameter(torch.ones(1, device=self.device))


   def optimize(self):
      # Mask the diagonal (dominant outlier) and half of
      # the matrix to not double-count the evidence.
      mask = torch.ones((self.sz,self.sz), device=self.device).triu(1)
      # Compute distances between loci.
      u = torch.ones((self.sz,self.sz), device=self.device).triu()
      dmat = torch.matrix_power(u, 2)
      dmat = dmat + torch.t(dmat)
      dmat[dmat < 1] = 0.0
      # Optimizer and scheduler.
      optim = torch.optim.Adam(self.parameters(), lr=.01)
      sched = torch.optim.lr_scheduler.MultiStepLR(optim,
            milestones=[3000])
      # Weights (mappability biases etc.)
      sqD = torch.sqrt(torch.diag(self.HiC.data))
      W = torch.ger(sqD,sqD)
      # Gradient descent proper.
      for step in range(3200):
         # Take the sigmoid to constrain 'P' within (0,1).
         P = torch.sigmoid(self.p)
         x = P*self.b - (1-P)*self.b
         AB = torch.ger(x,x)
         # Expected counts.
         mu = W * torch.exp(AB + self.g - dmat*self.a * .01)
         # Reparametrize for the negative binomial.
         nbp = mu / (mu + self.t)
         log_p = NegativeBinomial(self.t, nbp).log_prob(self.HiC.data)
         # Multiply by mask to remove entries.
         ll = -torch.sum(log_p * mask)
         optim.zero_grad()
         ll.backward()
         torch.nn.utils.clip_grad_norm_(self.parameters(), .01)
         optim.step()
         sched.step()
         #sys.stderr.write('%d %f\n' % (step, float(ll)))
         
      sys.stdout.write('# alpha %f\n' % float(self.a))
      sys.stdout.write('# beta %f\n' % float(self.b))
      sys.stdout.write('# gamma %f\n' % float(self.g))
      for i in range(self.sz):
         x = float(torch.sigmoid(self.p[i]))
         if abs(x - 0.73105857) > 1e-5:
            sys.stdout.write('%d\t%f\n' % (i, x))
         else:
            sys.stdout.write('%d\tNA\n' % i)

if __name__ == '__main__':
   sz = int(sys.argv.pop(1)) if sys.argv[1].isdigit() else 0
   sys.stderr.write('%s\n' % sys.argv[1])
   HiC = HiCData(path=sys.argv[1], sz=sz)
   for fname in sys.argv[2:]:
      sys.stderr.write('%s\n' % fname)
      HiC.add(fname)
   #HiC = torch.load(sys.argv[1])
   #HiC.show_data(sys.stdout)
   #torch.save(HiC, '13_ESH_mus.tch')
   M = Model(HiC)
   M.optimize()
