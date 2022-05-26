### code for this basline is copied directly from single.py in https://github.com/facebookresearch/GradientEpisodicMemory

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .common1 import *


def compute_offsets(task, nc_per_task, flag):
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

class Net(torch.nn.Module):

    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        nl, nh = args.n_layers, args.n_hiddens

        # setup network
        self.is_cifar = (args.data_file == 'cifar100.pt')
        self.is_tiered = (args.data_file == 'tieredimagenet.pt')
        if self.is_cifar:
            self.flag1 = True
            self.net = ResNetc(100,32,1024)
        if self.is_tiered:
            self.flag1=True
            self.net = ResNet18t(96,20,84)
        else:
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])

        # setup optimizer
        self.opt = torch.optim.SGD(self.parameters(), lr=args.lr)

        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()

        if self.is_cifar:
            self.nc_per_task = 10
        if self.is_tiered:
            self.nc_per_task = 4
        else:
            self.nc_per_task = n_outputs
        self.n_outputs = n_outputs
        # handle gpus if specified
        self.cuda = args.cuda
        if self.cuda:
            self.net = self.net.cuda()

    def forward(self, x, t):
        output = self.net(x,t)
        if self.is_cifar or self.is_tiered:
            # make sure we predict classes within the current task
            offset1, offset2 = compute_offsets(t,self.nc_per_task,self.flag1)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, t, y):
        self.train()
        self.zero_grad()
        offset1, offset2 = compute_offsets(t, self.nc_per_task, self.flag1)
        if self.is_cifar:
            self.bce((self.net(x,t)[:, offset1: offset2]),
                     y - offset1).backward()
        elif self.is_tiered:
            self.bce((self.net(x,t)[:, offset1: offset2]),
                     y).backward()
        else:
            self.bce(self.net(x, t), y).backward()
        self.opt.step()
