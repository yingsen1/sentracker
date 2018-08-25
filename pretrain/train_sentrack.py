#coding:utf-8

import os
import sys
import pickle
import time

import torch
import torch.optim as optim
from torch.autograd import Variable

from data_prov import *
from model import *
from train_config import *

# 处理一下路径
img_home = trainOpt.seq_home
data_path = trainOpt.output_path

# 优化器
def set_optimizer(model, lr_base, lr_mult=trainOpt.lr_mult, momentum=trainOpt.momentum, w_decay=trainOpt.w_decay):
    params = model.get_learnable_params()
    param_list = []
    for k, p in params.iteritems():
        lr = lr_base
        for l, m in lr_mult.iteritems(): # lr_mult为一个字典
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params':[p], 'lr':lr})
    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay) #TODO：这里需要加强理解
    return optimizer

# 训练主程序
def train_mdnet():
    ## 准备 dataset
    with open(data_path, 'rb') as fp:
        data = pickle.load(fp)

    K = len(data) # 序列的个数
    dataset = [None]*K # 创造K个元素的None列表, 作为存储K个视频集的迭代器的容器
    for k, (seqname, seq) in enumerate(data.iteritems()): #在prepro_data中，data字典{'seqname':seq};seq:{'images','gt'}
        img_list = seq['images'] # 只是序列文件夹里面的名字
        gt = seq['gt']
        img_dir = os.path.join(img_home, seqname)
        dataset[k] = RegionDataset(img_dir, img_list, gt)

    ## init model
    print(trainOpt.init_moedel_path)
    model = MDNet(trainOpt.init_moedel_path, K) # K个视频集对应K个分支
    if trainOpt.use_gpu:
        model = model.cuda()
    model.set_learnable_params(trainOpt.ft_layers)

    ## 初始化目标函数和优化器
    criterion = BinaryLoss() # TODO：可以尝试一下用FL
    evaluator = Precision() #TODO: 加强理解
    optimizer = set_optimizer(model, trainOpt.lr)

    best_prec = 0. # 初始化
    for i in range(trainOpt.n_cycles): # 定义epoch
        print('==== Start Cycle %d ===='%(i))
        k_list = np.random.permutation(K) # 随机训练第k%K个分支 # 每个epoch都不一样 #TODO：这个有影响吗？
        prec = np.zeros(K) # 保存精度的容器 #TODO: 可以用torchnet.meter代替
        for j, k in enumerate(k_list):
            tic = time.time()
            pos_regions, neg_regions = dataset[k].next()

            pos_regions = Variable(pos_regions) #TODO：这里为什么要Variable？？
            neg_regions = Variable(neg_regions)

            if trainOpt.use_gpu:
                pos_regions = pos_regions.cuda()
                neg_regions = neg_regions.cuda()

            pos_score = model(pos_regions, k)
            neg_score = model(neg_regions, k)

            loss = criterion(pos_score, neg_score)
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), trainOpt.grad_clip)
            optimizer.step()

            prec[k] = evaluator(pos_score, neg_score)

            toc = time.time()-tic
            print("Cycle %2d, K %2d (%2d), Loss %.3f, Prec %.3f, Time %.3f"%(i,j,k,loss.data[0],prec[k],toc))

        cur_prec = prec.mean()
        print('Mean Precision: %.3f'%(cur_prec))
        if cur_prec > best_prec: # 说明训练有效 #TODO: 不要完全覆盖，每50个epoch保存一次
            best_prec = cur_prec
            if trainOpt.use_gpu:  # 统一存到CPU上
                model = model.cpu()
            states = {'shared_layers': dict(model.layers.state_dict(), **model.seblocks.state_dict())} # 只存这部分的参数
            print("Save model to %s"%(trainOpt.model_path))
            torch.save(states, trainOpt.model_path)
            if trainOpt.use_gpu:
                model = model.cuda()

if __name__ == '__main__':
    train_mdnet()






