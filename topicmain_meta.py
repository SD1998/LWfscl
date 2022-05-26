#!/usr/bin/env python
# -*- coding:utf-8 -*-
import torch
import importlib
import datetime
import argparse
import random
import uuid
import time
import os
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from metrics.metrics import confusion_matrix
#
import os.path as osp
from copy import deepcopy
from dataloader.sampler import CategoriesSampler
from utilsdata import count_acc,Averager
from dataloader.data_utils import set_up_datasets,get_dataloader,get_base_dataloader,get_base_dataloader_meta,get_new_dataloader,get_session_classes

checkpoint_path = 'checkpoint/'
#
class Trainer():
    def __init__(self, model,args):
        self.args = args
        self.model = model
        self.args = set_up_datasets(self.args)
        self.testdataloaderlist=[]
    def get_dataloader(self,session):
        if session == 0:
            trainset, trainloader, testloader = self.get_base_dataloader_meta()
        else:
            trainset, trainloader, testloader = self.get_new_dataloader(session)
        return trainset, trainloader, testloader
    def get_base_dataloader_meta(self):
        txt_path = "data/index_list/" + self.args.dataset + "/session_" + str(0 + 1) + '.txt'
        class_index = np.arange(self.args.base_class)
        if self.args.dataset == 'cifar100':
            # class_index = np.arange(self.args.base_class)
            trainset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=True, download=True,
                                                  index=class_index, base_sess=True)
            testset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=False, download=False,
                                                 index=class_index, base_sess=True)

        if self.args.dataset == 'cub200':
            # class_index = np.arange(self.args.base_class)
            trainset = self.args.Dataset.CUB200(root=self.args.dataroot, train=True, index_path=txt_path)
            testset = self.args.Dataset.CUB200(root=self.args.dataroot, train=False, index=class_index)
        if self.args.dataset == 'mini_imagenet':
            trainset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=True, index_path=txt_path)
            testset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=False, index=class_index)

        # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
        #自定义取样本策略
        sampler = CategoriesSampler(trainset.targets, self.args.train_episode, self.args.episode_way,
                                    self.args.episode_shot + self.args.episode_query)

        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=8,
                                                  pin_memory=True)

        testloader = torch.utils.data.DataLoader(
            dataset=testset, batch_size=self.args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

        return trainset, trainloader, testloader

    def get_new_dataloader(self, session):
        txt_path = "data/index_list/" + self.args.dataset + "/session_" + str(session + 1) + '.txt'
        if self.args.dataset == 'cifar100':
            class_index = open(txt_path).read().splitlines()
            trainset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=True, download=False,
                                                  index=class_index, base_sess=False)
        if self.args.dataset == 'cub200':
            trainset = self.args.Dataset.CUB200(root=self.args.dataroot, train=True,
                                                index_path=txt_path)
        if self.args.dataset == 'mini_imagenet':
            trainset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=True,
                                                      index_path=txt_path)
        if self.args.batch_size_new == 0:
            batch_size_new = trainset.__len__()
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                      num_workers=8, pin_memory=True)
        else:
            trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=self.args.batch_size_new,
                                                      shuffle=True,
                                                      num_workers=8, pin_memory=True)

        class_new = self.get_session_classes(session)

        if self.args.dataset == 'cifar100':
            testset = self.args.Dataset.CIFAR100(root=self.args.dataroot, train=False, download=False,
                                                 index=class_new, base_sess=False)
        if self.args.dataset == 'cub200':
            testset = self.args.Dataset.CUB200(root=self.args.dataroot, train=False, index=class_new)
        if self.args.dataset == 'mini_imagenet':
            testset = self.args.Dataset.MiniImageNet(root=self.args.dataroot, train=False, index=class_new)

        testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=self.args.test_batch_size, shuffle=False,
                                                 num_workers=8, pin_memory=True)

        return trainset, trainloader, testloader
    def get_session_classes(self, session):
        if session==0:
            start=0
        else:
            start=self.args.base_class+(session-1)*self.args.way
        class_list = np.arange(start,self.args.base_class + session * self.args.way)
        return class_list
    def life_experience(self):
        result_a = []
        result_t = []

        current_task = 0
        time_start = time.time()
        print("training")
    #
        for session in range(self.args.sessions):
            result_m = 0
            t = session
            train_set, trainloader, testloader = self.get_dataloader(session)
            self.testdataloaderlist.append(testloader)
            if self.args.pretrain and session==0:
                print(self.model.state_dict())
                model_dict=torch.load(self.args.pretrain_model)
                self.model.load_state_dict(model_dict)
                result_tem = self.eval_tasks(session)
                print(result_tem)
            else:
                if session==0:
                    epochs=self.args.n_epochs
                else:
                    epochs=self.args.n_newepochs
                if session==1:
                    self.model.set_gamma(args.new_gamma)
                flag1=True
                for epoch in range(epochs):
                    if epoch==1:
                        flag1=False
                    if epoch>10 and t==0:
                        self.args.gamma = 0.05
                        self.model.set_gamma(0.05)
                    if epoch>15 and t==0:
                        self.args.gamma =0.01
                        self.model.set_gamma(0.01)
                    if epoch>30 and t==0:
                        self.args.gamma =0.005
                        self.model.set_gamma(0.005)
                    for (i, data) in enumerate(trainloader):
                        v_x,v_y = data
                        #print(v_x.size(),v_y.size())
                        '''
                        if (((i % args.log_every) == 0) or (t != current_task)):
                            result_a.append(eval_tasks(model, x_te, args))
                            result_t.append(current_task)
                            current_task = t
                        '''
                        v_y = v_y.long()

                        if args.cuda:
                            v_x = v_x.cuda()
                            v_y = v_y.cuda()
                        self.model.train()
                        self.model.observe(Variable(v_x), t, Variable(v_y),flag1)
                        torch.cuda.empty_cache()
                    result_tem = self.eval_tasks(session)
                    print(epoch,result_tem)
                    if result_tem > result_m:
                        result_m = result_tem
                    if epoch%20==0:
                        torch.save(self.model.state_dict(), checkpoint_path +'-'+ str("%d" % t) +'-'+ str("%d" % epoch) + '-' + str("%.4f" % result_tem) + '.pth.tar')

                print("testing")
                result_a.append(result_m)
                result_t.append(t)

        time_end = time.time()
        time_spent = time_end - time_start

        return torch.Tensor(result_t), torch.Tensor(result_a), time_spent

    def eval_tasks(self,session):
        self.model.eval()
        va = Averager()
        with torch.no_grad():
            for t,testloader in enumerate(self.testdataloaderlist):
                for i, batch in enumerate(testloader):
                    data, test_label = [_.cuda() for _ in batch]
                    y = test_label.data.cpu().numpy()
                    if t!=0:
                        y = (test_label-self.args.base_class)%self.args.way
                    y=Variable(torch.from_numpy(np.array(y))).long()
                    y=y.cuda()
                    logits =self.model(data,t)
                    acc = count_acc(logits, y)
                    va.add(acc)
            va = va.item()
        return va

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continuum learning')

    # model details
    parser.add_argument('--model', type=str, default='single',
                        help='model to train')
    parser.add_argument('--n_hiddens', type=int, default=100,
                        help='number of hidden neurons at each layer')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='number of hidden layers')
    parser.add_argument('--finetune', default='yes', type=str, help='whether to initialize nets in indep. nets')

    parser.add_argument('--n_tasks', type=int, default=2,
                        help='number of tasks')

    # optimizer parameters influencing all models
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs per task')
    parser.add_argument('--n_newepochs', type=int, default=1,
                        help='Number of epochs per new task')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='the amount of items received by the algorithm at one time (set to 1 across all experiments). Variable name is from GEM project.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--new_lr', type=float, default=1e-3,
                        help='new learning rate')
    parser.add_argument('-decay', type=float, default=0.0005)

    # memory parameters for GEM baselines
    parser.add_argument('--n_memories', type=int, default=0,
                        help='number of memories per task')
    parser.add_argument('--memory_strength', default=0, type=float,
                        help='memory strength (meaning depends on memory)')
    # memory parameters for OWM baselines
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='The degree of network forgetting')
    # memory parameters for FCL baselines
    parser.add_argument('--bound', type=int, default=1,
                        help='The bound of tasks')
    # parameters specific to models in https://openreview.net/pdf?id=B1gTShAct7

    parser.add_argument('--memories', type=int, default=5120,
                        help='number of total memories stored in a reservoir sampling based buffer')

    parser.add_argument('--gamma', type=float, default=1.0,
                        help='gamma learning rate parameter')  # gating net lr in roe
    parser.add_argument('--new_gamma', type=float, default=1.0,
                        help='gamma learning rate parameter')  # gating net lr in roe

    parser.add_argument('--batches_per_example', type=float, default=1,
                        help='the number of batch per incoming example')

    parser.add_argument('--s', type=float, default=1,
                        help='current example learning rate multiplier (s)')

    parser.add_argument('--replay_batch_size', type=float, default=20,
                        help='The batch size for experience replay. Denoted as k-1 in the paper.')

    parser.add_argument('--beta', type=float, default=1.0,
                        help='beta learning rate parameter')  # exploration factor in roe

    # experiment parameters
    parser.add_argument('--cuda', type=str, default='no',
                        help='Use GPU?')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed of model')
    parser.add_argument('--log_every', type=int, default=100,
                        help='frequency of logs, in minibatches')
    parser.add_argument('--save_path', type=str, default='results/',
                        help='save models at the end of training')

    # data parameters
    parser.add_argument('--data_path', default='data/',
                        help='path where data is located')
    parser.add_argument('--data_file', default='mnist_permutations.pt',
                        help='data file')
    parser.add_argument('--samples_per_task', type=int, default=-1,
                        help='training samples per task (all if negative)')
    parser.add_argument('--shuffle_tasks', type=str, default='no',
                        help='present tasks in order')

    #新添加的参数
    #数据集参数
    parser.add_argument('--dataset', type=str, default='cub200',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('--dataroot', type=str, default='data/')
    # for episode learning
    parser.add_argument('--train_episode', type=int, default=50)
    parser.add_argument('--episode_shot', type=int, default=1)
    parser.add_argument('--episode_way', type=int, default=15)
    parser.add_argument('--episode_query', type=int, default=15)

    parser.add_argument('--batch_size_new', type=int, default=10, help='set 0 will use all the availiable training image for new')
    parser.add_argument('--test_batch_size', type=int, default=100)
    parser.add_argument('--pretrain', type=str, default='no',
                        help='Use pretrain model?')
    parser.add_argument('--pretrain_model', type=str, default='checkpoint/model/')

    args = parser.parse_args()

    args.cuda = True if args.cuda == 'yes' else False
    args.finetune = True if args.finetune == 'yes' else False
    args.pretrain = True if args.pretrain =='yes' else False

    # taskinput model has one extra layer
    if args.model == 'taskinput':
        args.n_layers -= 1

    # unique identifier
    uid = uuid.uuid4().hex

    # initialize seeds
    torch.backends.cudnn.enabled = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        print("Found GPU:", torch.cuda.get_device_name(0))
        torch.cuda.manual_seed_all(args.seed)

    # load model
    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(args)
    if args.cuda:
        try:
            model.cuda()
        except:
            pass
    print("builing model")
    print(args)

    # run model on continuum
    train_tool =Trainer(model,args)
    result_t, result_a, spent_time = train_tool.life_experience()
    print("training and testing")

    # prepare saving path and file name
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    fname = args.model + '_' + args.data_file + '_'
    fname += datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fname += '_' + uid
    fname = os.path.join(args.save_path, fname)

    # save confusion matrix and print one line of stats
    stats = confusion_matrix(result_t, result_a, fname + '.txt')
    one_liner = str(vars(args)) + ' # '
    one_liner += ' '.join(["%.3f" % stat for stat in stats])
    print(fname + ': ' + one_liner + ' # ' + str(spent_time))

    # save all results in binary file
    torch.save((result_t, result_a, model.state_dict(),
                stats, one_liner, args), fname + '.pt')
