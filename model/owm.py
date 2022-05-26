#OWM算法的更新在每一层网络层

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

from .common1 import MLP, ResNet18
import random
from torch.nn.modules.loss import CrossEntropyLoss
from random import shuffle
import sys
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")


class Net(nn.Module):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 n_tasks,
                 args):
        super(Net, self).__init__()
        
        nl, nh = args.n_layers, args.n_hiddens

        self.alpha = args.alpha
        self.lr=args.lr
        self.net = MLP([n_inputs] + [nh] * nl + [n_outputs], self.alpha)


        self.bce = CrossEntropyLoss()
        self.n_outputs = n_outputs

        # allocate buffer
        self.age = 0
        
        # handle gpus if specified
        self.cuda = args.cuda
        if self.cuda:
            self.net = self.net.cuda()

    # 输入穿过一次网络获得结果
    def forward(self, x, t,flag=False):
        output = self.net(x,t,flag)
        return output

                

    def observe(self, x, t, y):
        ### step through elements of x
        #before = deepcopy(self.net.state_dict())
        self.net.zero_grad()

        # handle gpus if specified
        loss = 0.0
        prediction = self.forward(x,t,False)
        loss = self.bce(prediction,y)
        loss.backward()
        self.net.opt(self.lr)
        prediction = self.forward(x, t,True)



