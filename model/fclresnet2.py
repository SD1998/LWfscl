#!/usr/bin/env python
# -*- coding:utf-8 -*-
# An implementation of Meta-Experience Replay (MER) Algorithm 7 from https://openreview.net/pdf?id=B1gTShAct7

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 修改关键：1、输出多头2、卷积神经网络

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from .common1 import *
from .Resnet import *
import random
from torch.nn.modules.loss import CrossEntropyLoss
from random import shuffle
import sys
from copy import deepcopy
import warnings
from random import shuffle

torch.cuda.set_device(0)
warnings.filterwarnings("ignore")


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.args = args

        self.lr = args.new_lr

        self.is_cub = (args.dataset == 'cub200')
        self.is_mini = (args.dataset == 'mini_imagenet')
        nl, nh = args.n_layers, args.n_hiddens
        #
        if self.is_cub:
            self.nc_base = 100
            self.nc_per_task = 10
            self.n_class = 200
            self.encoder = resnet18(True,
                                    self.n_class)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512
        elif self.is_mini:
            self.nc_base = 60
            self.nc_per_task = 5
            self.n_class = 100
            self.encoder = resnet18(False, self.n_class)  # pretrained=False
            self.num_features = 512
        self.classifer = MLP([512] + [nh] * nl + [self.n_class], args.alpha, args.gamma)

        self.bound = args.bound

        self.bce = CrossEntropyLoss()

        self.opt = optim.SGD(self.parameters(), args.lr, momentum=0.9, nesterov=True, weight_decay=args.decay)
        self.batchSize = int(args.replay_batch_size)

        self.memories = args.memories
        self.s = float(args.s)
        self.gamma = args.gamma
        self.flag = True
        self.flag1 = True

        # allocate buffer
        self.M = []
        self.age = 0

        # handle gpus if specified
        self.cuda = args.cuda
        if self.cuda:
            self.encoder = self.encoder.cuda()
            self.classifer = self.classifer.cuda()

    def set_gamma(self, gamma):
        self.classifer.set_gamma(gamma)

    def compute_offsets(self, task):
        """
            Compute offsets for cifar to determine which
            outputs to select for a given task.
        """
        if task == 0:
            offset1 = 0
        else:
            offset1 = (task - 1) * self.nc_per_task + self.nc_base
        offset2 = task * self.nc_per_task + self.nc_base
        return offset1, offset2

    # 输入穿过一次网络获得结果
    def forward(self, x, t, flag=False):
        x = self.encoder(x)
        output = self.classifer(x, t, flag)
        # 多头处理，截取任务对应输出
        # make sure we predict classes within the current task
        offset1, offset2 = self.compute_offsets(t)
        offset1 = int(offset1)
        offset2 = int(offset2)
        if offset1 > 0:
            output[:, :offset1].data.fill_(-10e10)
        if offset2 < self.n_class:
            output[:, offset2:self.n_class].data.fill_(-10e10)

        return output[:, offset1: offset2]

    def getBatch(self, x, y, t):
        mxi = Variable(torch.from_numpy(np.array(x))).float().view(1, 3, 224, 224)
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
                xi = Variable(torch.from_numpy(np.array(x))).float().view(1, 3, 224, 224)
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

        return bx2, by2, bt2, myindex

    def observe(self, x, t, y, flags):
        if t != 0 and self.flag1:
            for param in self.encoder.parameters():  # nn.Module有成员函数parameters()
                param.requires_grad = False
            self.opt = optim.SGD(self.classifer.parameters(), self.lr)
            self.flag1 = False
        ### step through elements of x
        if t % self.bound == 1 and self.flag:
            print(t)
            self.classifer.reserve_p()
            # review
            self.age = 0
            self.M = []
            # review
            self.flag = False
        if t % self.bound != 1:
            self.flag = True
        if t == 0:
            if self.flag1:
                weights_before1 = deepcopy(self.encoder.state_dict())
            self.encoder.zero_grad()
            self.classifer.zero_grad()
            weights_before = deepcopy(self.classifer.state_dict())
            self.encoder.zero_grad()
            self.classifer.zero_grad()
            offset1, offset2 = self.compute_offsets(t)
            prediction = self.forward(x, t, False)
            loss = self.bce(prediction, y - offset1)
            loss.backward()
            self.opt.step()
            if self.flag1:
                weights_after1 = self.encoder.state_dict()
            weights_after = self.classifer.state_dict()
            # Reptile meta-update
            if self.flag1:
                print(self.args.gamma)
                self.encoder.load_state_dict(
                    {name: weights_before1[name] + ((weights_after1[name] - weights_before1[name]) * self.args.gamma)
                     for
                     name in weights_before1})
            self.classifer.meta_learn(weights_before, weights_after)
        else:
            for i in range(0, x.size()[0]):
                self.age += 1
                xi = x[i].data.cpu().numpy()
                yi = y[i].data.cpu().numpy()
                torch.cuda.empty_cache()
                if self.age > 1:
                    if self.flag1:
                        weights_before1 = deepcopy(self.encoder.state_dict())
                    self.encoder.zero_grad()
                    self.classifer.zero_grad()
                    weights_before = deepcopy(self.classifer.state_dict())

                    # Draw batch from buffer:
                    bxs, bys, bts, myind = self.getBatch(xi, yi, t)

                    # SGD on individual samples from batch:
                    loss = 0.0
                    for idx in range(len(bxs)):
                        # if self.flag1:
                        self.encoder.zero_grad()
                        self.classifer.zero_grad()
                        bx = bxs[idx]
                        by = bys[idx]
                        bt = bts[idx]
                        # 截取输出
                        offset1, offset2 = self.compute_offsets(bt)
                        prediction = self.forward(bx, bt, False)
                        # print(prediction,by-offset1,by)
                        if myind == idx:
                            # High learning rate SGD on current example:
                            loss = self.s * self.bce(prediction, by - offset1)
                        else:
                            loss = self.bce(prediction, by - offset1)

                        loss.backward()
                        self.opt.step()

                    if self.flag1:
                        weights_after1 = self.encoder.state_dict()
                    weights_after = self.classifer.state_dict()
                    # Reptile meta-update
                    if self.flag1:
                        self.encoder.load_state_dict({name: weights_before1[name] + (
                        (weights_after1[name] - weights_before1[name]) * self.args.new_gamma) for name in
                                                      weights_before1})
                    self.classifer.meta_learn(weights_before, weights_after)

                sys.stdout.flush()

                # Reservoir sampling memory update:
                if flags:
                    if len(self.M) < self.memories:
                        self.M.append([xi, yi, t])

                    else:
                        p = random.randint(0, self.age)
                        if p < self.memories:
                            self.M[p] = [xi, yi, t]
        if flags:
            with torch.no_grad():
                prediction = self.forward(x, t, True)

