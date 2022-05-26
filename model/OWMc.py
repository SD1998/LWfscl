# OWM算法的更新在每一层网络层

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from .common1 import MLP, ResNet18,ResNetc
import random
from torch.nn.modules.loss import CrossEntropyLoss
from random import shuffle
import sys
from copy import deepcopy
import warnings

warnings.filterwarnings("ignore")

def compute_offsets(task, nc_per_task, is_cifar):
    """
        Compute offsets for cifar to determine which
        outputs to select for a given task.
    """
    if is_cifar:
        offset1 = task * nc_per_task
        offset2 = (task + 1) * nc_per_task
    else:
        offset1 = 0
        offset2 = nc_per_task
    return offset1, offset2

class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()

        nl, nh = args.n_layers, args.n_hiddens
        self.is_cifar = (args.data_file == 'cifar100.pt')
        self.is_tiered = (args.data_file == 'tieredimagenet.pt')
        self.alpha = args.alpha
        self.lr = args.lr
        if self.is_cifar:
            self.net = ResNetc(10,self.alpha)
        elif self.is_tiered:
            self.flag1 = True
            self.net = ResNetc(96, 84, 18496, True, self.alpha, args.gamma)
        else:
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs], self.alpha)

        self.bce = CrossEntropyLoss()
        self.n_outputs = n_outputs

        # allocate buffer
        self.age = 0

        if self.is_cifar:
            self.nc_per_task = 10
        elif self.is_tiered:
            self.nc_per_task = 4
        else:
            self.nc_per_task = n_outputs

        # handle gpus if specified
        self.cuda = args.cuda
        if self.cuda:
            self.net = self.net.cuda()

    # 输入穿过一次网络获得结果
    def forward(self, x, t, flag=False):
        output = self.net(x, t, flag)
        if self.is_cifar or self.is_tiered:
            # make sure we predict classes within the current task
            offset1 = int(t * self.nc_per_task)
            offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        return output

    def observe(self, x, t, y):
        ### step through elements of x
        # before = deepcopy(self.net.state_dict())
        self.net.zero_grad()

        # handle gpus if specified
        loss = 0.0
        offset1, offset2 = compute_offsets(t, self.nc_per_task, self.is_cifar)
        prediction = self.forward(x, t, False)[:, offset1: offset2]
        if self.is_cifar:
            loss = self.bce(prediction, y % self.nc_per_task)
        else:
            loss = self.bce(prediction, y)
        loss.backward()
        self.net.opt(self.lr)
        prediction = self.forward(x, t, True)
