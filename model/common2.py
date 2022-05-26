### A copy of common.py from https://github.com/facebookresearch/GradientEpisodicMemory.
### We leveraged the same architecture and weight initialization for all of our experiments.

# Copyright 2019-present, IBM Research
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d
#修改部分
from torch.autograd import Variable
from copy import deepcopy
dtype = torch.cuda.FloatTensor  # run on GPU


def deepcopy_list(x):
    y = []
    # 遍历列表，针对列表中的每一元素进行类型判断，并调用对应的复制函数,递归
    # 后续可以将方法值化，减少属性访问的时间，这是一个优化小细节
    for a in x:
        y.append(deepcopy(a))
    return y

class OWMLayer:

    def __init__(self,  shape, alpha=0):

        self.input_size = shape[0]
        self.output_size = shape[1]
        self.alpha = alpha
        self.P = Variable((1.0/self.alpha)*torch.eye(self.input_size).type(dtype), volatile=True)

    def force_learn(self, w, input_, learning_rate, alpha=1.0):  # input_(batch,input_size)
        self.r = torch.mean(input_, 0, True)
        self.k = torch.mm(self.P, torch.t(self.r))
        self.c = 1.0 / (alpha + torch.mm(self.r, self.k))  # 1X1
        self.P.sub_(self.c*torch.mm(self.k, torch.t(self.k)))
        w.data -= learning_rate * torch.mm(self.P.data, w.grad.data)
        w.grad.data.zero_()

    def predit_lable(self, input_, w,):
        return torch.mm(input_, w)
#修改部分

def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self, sizes, alpha=0,gamma=1.0):
        super(MLP, self).__init__()
        layers = []
        self.out = []
        self.si = 0
        self.Pi = []
        self.Pi1=[]
        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            self.si += 1
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())
        #存放每层的投影算子
        for j in range(0, self.si):
            P = Variable((1.0/1.0)*torch.eye(sizes[j]).type(dtype), volatile=True)
            self.Pi.append(P)
            self.Pi1.append(P.data)
        self.alpha_array = []
        self.gamma =gamma

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)
#计算衰减参数
    def fix(self,i,batch_size):
        lamda = i / (20000 // batch_size)
        self.alpha_array = [1.0 * 0.0001 ** lamda, 1.0 * 0.1 ** lamda, 0.5]
#保留投影算子大小
    def reserve_p(self):
        for i in range(0,len(self.Pi)):
            self.Pi1[i]=deepcopy(self.Pi[i].data)

    def meta_learn(self,before,after):
        outcome={}
        for name, i in zip(before, range(0,len(before))):
            if i%2==0:
               outcome[name]=before[name] + torch.mm((after[name] - before[name]),self.Pi1[int(i/2)]) * self.gamma
            else:
               outcome[name]=before[name] + (after[name] - before[name]) * self.gamma
        self.load_state_dict(outcome)
        #self.load_state_dict({name: before[name] + (torch.mm((after[name] - before[name]),self.Pi1[i]) * self.gamma) for name, i in zip(before, range(0,len(self.Pi1)))})

    # 计算更新投影算子
    def force_learn(self, input_,num=0):  # input_(batch,input_size)
        #self.Pi1 = deepcopy_list(self.Pi)
        #print(self.Pi1)
        self.r = torch.mean(input_, 0, True)
        self.k = torch.mm(self.Pi[num], torch.t(self.r))
        self.c = 1.0 / (self.alpha_array[num] + torch.mm(self.r, self.k))  # 1X1
        self.Pi[num].sub_(self.c*torch.mm(self.k, torch.t(self.k)))

#输入穿过一次网络获得结果
    def forward(self, x,t,flag=False):
        self.out.clear()
        n=0
        for i in range(0, len(self.net)):
            if i % 2 == 0 and flag:
                self.out.append(x)
                self.force_learn(x,n)
                n+=1
            x = self.net[i](x)
        return x
    def opt(self, lr):
        for i in range(0,len(self.Pi)):
            self.net[i*2].weight.data-=lr*torch.mm(self.net[i*2].weight.grad.data,self.Pi[i].data)
            self.net[i*2].weight.grad.data.zero_()



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf

        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(nclasses, nf=20):
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)
