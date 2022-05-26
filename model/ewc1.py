### We use the same version of EwC https://www.pnas.org/content/114/13/3521 originally used in https://github.com/facebookresearch/GradientEpisodicMemory
### We directly copied the ewc.py model file from the GEM project https://github.com/facebookresearch/GradientEpisodicMemory

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.autograd import Variable
from .common1 import *


class Net(torch.nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens
        self.reg = args.memory_strength

        # setup network
        self.is_cifar = (args.data_file == 'cifar100.pt')
        self.is_tiered = (args.data_file == 'tieredimagenet.pt')
        self.is_office = (args.data_file == 'office.pt')
        self.is_fru =(args.data_file=='fru.pt')
        if self.is_cifar:
            self.net = ResNet18(n_outputs)
        elif self.is_tiered:
            self.flag1 = True
            self.net = ResNetc(96,84,18496,False)
        elif self.is_office:
            self.flag1 = True
            self.net = ResNetc(120,256,230400,False)
        elif self.is_fru:
            self.flag1=True
            self.net = ResNetc(50,256,230400,False)
        else:
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])

        # setup optimizer
        self.opt = torch.optim.SGD(self.net.parameters(), lr=args.lr)

        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()

        # setup memories
        self.current_task = 0
        self.fisher = {}
        self.optpar = {}
        self.memx = None
        self.memy = None

        if self.is_cifar:
            self.nc_per_task = n_outputs / n_tasks
        elif self.is_tiered:
            self.nc_per_task = 4
        elif self.is_office:
            self.nc_per_task = 10
        elif self.is_fru:
            self.nc_per_task =5
        else:
            self.nc_per_task = n_outputs
        self.n_outputs = n_outputs
        self.n_memories = args.n_memories

        # handle gpus if specified
        self.cuda = args.cuda
        if self.cuda:
            self.net = self.net.cuda()

    def compute_offsets(self,task, nc_per_task, flag=False):
        """
            Compute offsets for cifar to determine which
            outputs to select for a given task.
        """
        if flag:
            offset1 = task * nc_per_task
            offset2 = (task + 1) * nc_per_task
        else:
            offset1 = 0
            offset2 = nc_per_task
        return offset1, offset2

    def forward(self, x, t):
        output = self.net(x)
        if self.is_cifar or self.is_tiered or self.is_office or self.is_fru:
            # make sure we predict classes within the current task
            offset1, offset2 = self.compute_offsets(t,self.nc_per_task,self.flag1)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, int(offset2):self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, t, y):
        self.net.train()

        # next task?
        if t != self.current_task:
            self.net.zero_grad()

            if self.is_cifar:
                offset1, offset2 = self.compute_offsets(self.current_task,self.nc_per_task,self.flag1)
                self.bce((self.net(Variable(self.memx))[:, offset1: offset2]),
                         Variable(self.memy) - offset1).backward()
            elif self.is_tiered or self.is_office or self.is_fru:
                offset1, offset2 = self.compute_offsets(self.current_task,self.nc_per_task,self.flag1)
                self.bce((self.net(Variable(self.memx))[:, offset1: offset2]),
                         Variable(self.memy)).backward()
            else:
                self.bce(self(Variable(self.memx),
                              self.current_task),
                         Variable(self.memy)).backward()
            self.fisher[self.current_task] = []
            self.optpar[self.current_task] = []
            for p in self.net.parameters():
                pd = p.data.clone()
                pg = p.grad.data.clone().pow(2)
                self.optpar[self.current_task].append(pd)
                self.fisher[self.current_task].append(pg)
            self.current_task = t
            self.memx = None
            self.memy = None

        if self.memx is None:
            self.memx = x.data.clone()
            self.memy = y.data.clone()
        else:
            if self.memx.size(0) < self.n_memories:
                self.memx = torch.cat((self.memx, x.data.clone()))
                self.memy = torch.cat((self.memy, y.data.clone()))
                if self.memx.size(0) > self.n_memories:
                    self.memx = self.memx[:self.n_memories]
                    self.memy = self.memy[:self.n_memories]

        self.net.zero_grad()
        if self.is_cifar:
            offset1, offset2 = self.compute_offsets(t)
            loss = self.bce((self.net(x)[:, offset1: offset2]),
                            y - offset1)
        elif self.is_tiered or self.is_office or self.is_fru:
            offset1, offset2 = self.compute_offsets(t,self.nc_per_task,self.flag1)
            loss = self.bce((self.net(x)[:, offset1: offset2]),y)
        else:
            loss = self.bce(self(x, t), y)
        for tt in range(t):
            for i, p in enumerate(self.net.parameters()):
                l = self.reg * Variable(self.fisher[tt][i])
                l = l * (p - Variable(self.optpar[tt][i])).pow(2)
                loss += l.sum()
        loss.backward()
        self.opt.step()
