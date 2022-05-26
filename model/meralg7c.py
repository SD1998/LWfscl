# An implementation of Meta-Experience Replay (MER) Algorithm 7 from https://openreview.net/pdf?id=B1gTShAct7

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from .common import MLP, ResNet18,ResNetc
import random
from torch.nn.modules.loss import CrossEntropyLoss
from random import shuffle
import sys
from copy import deepcopy
import warnings
from random import shuffle

torch.cuda.set_device(0)
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
        #
        self.is_cifar = (args.data_file == 'cifar100.pt')
        self.is_tiered =(args.data_file == 'tieredimagenet.pt')
        self.is_office = (args.data_file == 'office.pt')
        self.is_fru =(args.data_file== 'fru.pt')
        if self.is_cifar:
            self.flag1=True
            self.net = ResNetc(100,32,1024)
        elif self.is_tiered:
            self.flag1=True
            self.net = ResNetc(96,84,18496)
        elif self.is_office:
            self.flag1=True
            self.net =ResNetc(120,256,230400)
        elif self.is_fru:
            self.flag1=True
            self.net =ResNetc(50, 256, 230400)
        else:
            self.flag1=False
            self.net = MLP([n_inputs] + [nh] * nl + [n_outputs])

        self.bce = CrossEntropyLoss()
        self.n_outputs = n_outputs

        self.opt = optim.SGD(self.parameters(), args.lr)
        self.batchSize = int(args.replay_batch_size)

        self.memories = args.memories
        self.s = float(args.s)
        self.gamma = args.gamma

        # allocate buffer
        self.M = []
        self.age = 0
        #
        if self.is_cifar:
            self.nc_per_task = 10
        elif self.is_tiered:
            self.nc_per_task = 4
        elif self.is_office:
            self.nc_per_task =10
        elif self.is_fru:
            self.nc_per_task=5
        else:
            self.nc_per_task = n_outputs
        #

        # handle gpus if specified
        self.cuda = args.cuda
        if self.cuda:
            self.net = self.net.cuda()

    def forward(self, x, t):
        output = self.net(x)
        #多头处理，截取任务对应输出
        if self.is_cifar or self.is_tiered or self.is_office or self.is_fru:
            # make sure we predict classes within the current task
            offset1 = int(t * self.nc_per_task)
            offset2 = int((t + 1) * self.nc_per_task)
            if offset1 > 0:
                output[:, :offset1].data.fill_(-10e10)
            if offset2 < self.n_outputs:
                output[:, offset2:self.n_outputs].data.fill_(-10e10)
        #
        return output

    def getBatch(self, x, y, t):
        mxi = Variable(torch.from_numpy(np.array(x))).float().view(1, -1)
        myi = Variable(torch.from_numpy(np.array(y))).long().view(1)
        if self.cuda:
            mxi = mxi.cuda()
            myi = myi.cuda()
        bxs = [mxi]
        bys = [myi]
        bts = [t]

        if len(self.M) > 0:
            order = [i for i in range(0, len(self.M))]
            osize = min(self.batchSize, len(self.M))
            for j in range(0, osize):
                shuffle(order)
                k = order[j]
                x, y, t = self.M[k]
                xi = Variable(torch.from_numpy(np.array(x))).float().view(1, -1)
                yi = Variable(torch.from_numpy(np.array(y))).long().view(1)

                if self.cuda:
                    xi = xi.cuda()
                    yi = yi.cuda()

                bxs.append(xi)
                bys.append(yi)
                bts.append(t)

        bx2 = []
        by2 = []
        bt2 = []
        indexes = [ind for ind in range(len(bxs))]
        shuffle(indexes)
        for index in indexes:
            if index == 0:
                myindex = len(bx2)
            bx2.append(bxs[index])
            by2.append(bys[index])
            bt2.append(bts[index])


        return bx2, by2, bt2,myindex

    def observe(self, x, t, y):
        ### step through elements of x
        for i in range(0, x.size()[0]):
            self.age += 1
            xi = x[i].data.cpu().numpy()
            yi = y[i].data.cpu().numpy()
            if self.age > 1:
                self.net.zero_grad()

                weights_before = deepcopy(self.net.state_dict())

                # Draw batch from buffer:
                bxs, bys, bts,myind = self.getBatch(xi, yi, t)

                # SGD on individual samples from batch:
                loss = 0.0
                for idx in range(len(bxs)):
                    self.net.zero_grad()
                    bx = bxs[idx]
                    by = bys[idx]
                    bt = bts[idx]
                    # 截取输出
                    offset1, offset2 = compute_offsets(bt, self.nc_per_task, self.flag1)
                    prediction = self.forward(bx, bt)[:, offset1: offset2]
                    if myind == idx:
                        # High learning rate SGD on current example:
                        if self.is_cifar:
                            loss = self.s * self.bce(prediction, by%self.nc_per_task)
                        else:
                            loss = self.s * self.bce(prediction, by)
                    else:
                        if self.is_cifar:
                           loss = self.bce(prediction, by%self.nc_per_task)
                        else:
                            loss = self.bce(prediction, by)
                    loss.backward()
                    self.opt.step()

                weights_after = self.net.state_dict()
                # Reptile meta-update:
                self.net.load_state_dict(
                    {name: weights_before[name] + ((weights_after[name] - weights_before[name]) * self.gamma) for name
                     in weights_before})

            sys.stdout.flush()

            # Reservoir sampling memory update:
            if len(self.M) < self.memories:
                self.M.append([xi, yi, t])

            else:
                p = random.randint(0, self.age)
                if p < self.memories:
                    self.M[p] = [xi, yi, t]