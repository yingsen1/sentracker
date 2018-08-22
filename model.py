#coding:utf-8
import os
import scipy.io
import numpy as np
from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch


def append_params(params, module, prefix):
    '''
    添加网络的参数
    :param params: 需要添加得参数
    :param module: 所属于的nn.Sequential()
    :param prefix: 前缀(name)
    :return: None
    '''
    for child in module.children(): # children():返回一个最近子模块的iterator
        for k, p in child._parameters.iteritems(): # ?._parameters; what is k?
            if p is None: continue

            if isinstance(child, nn.BatchNorm2d):
                name = prefix + '_bn_' + k
            else:
                name = prefix + '_' + k

            if name not in params:
                params[name] = p
            else:
                raise RuntimeError("Duplicated param name: %s" % (name))


class LRN(nn.Module):
    def __init__(self):
        super(LRN, self).__init__()

    def forward(self, x):
        #
        # x: N x C x H x W
        pad = Variable(x.data.new(x.size(0), 1, 1, x.size(2), x.size(3)).zero_())
        x_sq = (x ** 2).unsqueeze(dim=1)
        x_tile = torch.cat((torch.cat((x_sq, pad, pad, pad, pad), 2),
                            torch.cat((pad, x_sq, pad, pad, pad), 2),
                            torch.cat((pad, pad, x_sq, pad, pad), 2),
                            torch.cat((pad, pad, pad, x_sq, pad), 2),
                            torch.cat((pad, pad, pad, pad, x_sq), 2)), 1)
        x_sumsq = x_tile.sum(dim=1).squeeze(dim=1)[:, 2:-2, :, :]
        x = x / ((2. + 0.0001 * x_sumsq) ** 0.75)
        return x


class MDNet(nn.Module):
    def __init__(self, model_path=None, K=1):
        super(MDNet, self).__init__()
        self.K = K # fc6分支的数量
        # 前面5层，con1、con2、con3、fc4、fc5
        self.layers = nn.Sequential(OrderedDict([
            ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                    nn.ReLU(),
                                    LRN(),
                                    nn.MaxPool2d(kernel_size=3, stride=2))),
            ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                    nn.ReLU())),
            ('fc4', nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(512 * 3 * 3, 512),
                                  nn.ReLU())),
            ('fc5', nn.Sequential(nn.Dropout(0.5),
                                  nn.Linear(512, 512),
                                  nn.ReLU()))]))
        # fc6,数量根据K来定
        self.branches = nn.ModuleList([nn.Sequential(nn.Dropout(0.5),
                                                     nn.Linear(512, 2)) for _ in range(K)])

        if model_path is not None:
            if os.path.splitext(model_path)[1] == '.pth':
                self.load_model(model_path)
            elif os.path.splitext(model_path)[1] == '.mat':
                self.load_mat_model(model_path)
            else:
                raise RuntimeError("Unkown model format: %s" % (model_path))
        self.build_param_dict()

    def build_param_dict(self):
        self.params = OrderedDict()
        for name, module in self.layers.named_children():
            append_params(self.params, module, name)
        for k, module in enumerate(self.branches): # branches:ModuleList()
            append_params(self.params, module, 'fc6_%d' % (k))

    def set_learnable_params(self, layers):
        '''
        控制需要学习的参数
        :param layers: 设置参数为可学习的层
        :return: None
        '''
        for k, p in self.params.iteritems(): # what is k??
            if any([k.startswith(l) for l in layers]): #any():所以为True则为True，反之
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        '''
        获得需要训练的参数
        :return: 返回需要训练的参数
        '''
        params = OrderedDict()
        for k, p in self.params.iteritems():
            if p.requires_grad:
                params[k] = p
        return params

    def forward(self, x, k=0, in_layer='conv1', out_layer='fc6'):
        '''
        前向传播，默认conv1->fc6，默认从第一个分支开始遍历
        :param x: input
        :param k: amount of branches
        :param in_layer: the first layer
        :param out_layer: the output layer
        :return: 2 points
        '''
        run = False # the signal to control the forward

        # conv1->fc5中运行
        for name, module in self.layers.named_children():
            if name == in_layer: # 判断in_layer是否在模型里边，从而控制forward
                run = True
            if run:
                x = module(x)
                if name == 'conv3': #更改conv3的形状，相当于flatten()
                    x = x.view(x.size(0), -1) # size()[0] is what? maybe Nbatch?
                if name == out_layer:
                    return x

        x = self.branches[k](x)
        if out_layer == 'fc6':
            return x
        elif out_layer == 'fc6_softmax':
            return F.softmax(x)

    def load_model(self, model_path):
        states = torch.load(model_path)
        shared_layers = states['shared_layers']
        self.layers.load_state_dict(shared_layers) # load layers' parameters

    def load_mat_model(self, matfile):
        print(matfile)
        mat = scipy.io.loadmat(matfile)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            self.layers[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.layers[i][0].bias.data = torch.from_numpy(bias[:, 0])


class BinaryLoss(nn.Module):
    def __init__(self):
        super(BinaryLoss, self).__init__()

    def forward(self, pos_score, neg_score):
        '''
        分别计算pos_loss和neg_loss，最后相加做一个二分类的softmax交叉熵
        :param pos_score: Nx(-score, +score)
        :param neg_score: Nx(-score, +score)
        :return: 交叉熵
        '''
        pos_loss = -F.log_softmax(pos_score, dim=1)[:, 1]  # y轴进行运算 #第1列为正得分
        neg_loss = -F.log_softmax(neg_score, dim=1)[:, 0]  # 第0列为负得分

        loss = pos_loss.sum() + neg_loss.sum() # 参照公式
        return loss


class Accuracy():
    '''
    自定义的accuracy，正确判断前景背景的概率
    '''
    def __call__(self, pos_score, neg_score):
        pos_correct = (pos_score[:, 1] > pos_score[:, 0]).sum().float() #正样本得到的得分，记录正确判断为前背景的次数
        neg_correct = (neg_score[:, 1] < neg_score[:, 0]).sum().float() #负样本得到的得分，记录正确判断为负背景的次数

        pos_acc = pos_correct / (pos_score.size(0) + 1e-8) # 1e-8为了防止数据出错/溢出
        neg_acc = neg_correct / (neg_score.size(0) + 1e-8)

        return pos_acc.data[0], neg_acc.data[0] # 都为Variable变量


class Precision():
    '''
    不太明白含义
    '''
    def __call__(self, pos_score, neg_score):
        scores = torch.cat((pos_score[:, 1], neg_score[:, 1]), 0)# 包含正样本和负样本得到的正得分
        topk = torch.topk(scores, pos_score.size(0))[1] # 保存的是topk的索引，（包含正负样本得到的正得分）
        prec = (topk < pos_score.size(0)).float().sum() / (pos_score.size(0) + 1e-8)

        return prec.data[0]
