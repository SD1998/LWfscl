### code for this basline is copied directly from independent.py in https://github.com/facebookresearch/GradientEpisodicMemory

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
        self.nets = torch.nn.ModuleList()
        self.opts = []

        self.is_cifar = (args.data_file == 'cifar100.pt')
        self.is_tiered = (args.data_file == 'tieredimagenet.pt')
        self.is_office = (args.data_file == 'office.pt')
        self.is_fru = (args.data_file == 'fru.pt')
        if self.is_cifar:
            self.nc_per_task = 10
        elif self.is_tiered:
            self.nc_per_task = 4
        elif self.is_office:
            self.nc_per_task = 10
        elif self.is_fru:
            self.nc_per_task =5
        self.n_outputs = n_outputs

        # setup network
        print(n_tasks)
        for _ in range(n_tasks):
            if self.is_cifar:
                self.nets.append(
                    ResNet18(int(n_outputs / n_tasks), int(20 / n_tasks)))
            elif self.is_tiered:
                self.n_outputs=96
                self.nets.append(
                    ResNetc(4, 84, 18496,False))
            elif self.is_office:
                self.n_outputs=120
                self.nets.append(
                    ResNetc(10,256,230400,False))
            elif self.is_fru:
                self.n_outputs=50
                self.nets.append(
                    ResNetc(10,256,230400,False))
            else:
                self.nets.append(
                    MLP([n_inputs] + [int(nh / n_tasks)] * nl + [n_outputs]))

        # setup optimizer
        for t in range(n_tasks):
            self.opts.append(torch.optim.SGD(self.nets[t].parameters(),
                                             lr=args.lr))

        # setup loss
        self.bce = torch.nn.CrossEntropyLoss()

        self.finetune = args.finetune
        self.gpu = args.cuda
        self.old_task = 0

    def forward(self, x, t):
        output = self.nets[t](x)
        if self.is_cifar or self.is_tiered or self.is_office or self.is_fru:
            bigoutput = torch.Tensor(x.size(0), self.n_outputs)
            if self.gpu:
                bigoutput = bigoutput.cuda()
            bigoutput.fill_(-10e10)
            bigoutput[:, int(t * self.nc_per_task): int((t + 1) * self.nc_per_task)].copy_(
                output.data)
            return Variable(bigoutput)
        else:
            return output

    def observe(self, x, t, y):
        # detect beginning of a new task
        if self.finetune and t > 0 and t != self.old_task:
            # initialize current network like the previous one
            for ppold, ppnew in zip(self.nets[self.old_task].parameters(),
                                    self.nets[t].parameters()):
                ppnew.data.copy_(ppold.data)
            self.old_task = t

        self.train()
        self.zero_grad()
        if self.is_cifar:
            self.bce(self.nets[t](x), y - int(t * self.nc_per_task)).backward()
        elif self.is_tiered or self.is_office or self.is_fru:
            self.bce(self.nets[t](x), y).backward()
        else:
            self.bce(self(x, t), y).backward()
        self.opts[t].step()
