#coding:utf-8
import numpy as np
import os
import sys
import time
import argparse
import json
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable


from sample_generator import *
from data_prov import *
from model import *
from bbreg import *
from gen_config import *
from config import *

np.random.seed(123) # 让每次算法起始阶段的随机数产生相同
torch.manual_seed(456)
torch.cuda.manual_seed(789)


def forward_samples(model, image, samples, out_layer='conv3'):
    '''
    将image_patch（切分过）输入网络并提取out_layer层的feature map
    :param model: 模型
    :param image: 图像块
    :param samples: samples的坐标
    :param out_layer: 层
    :return: 指定的feature map
    '''
    model.eval()
    extractor = RegionExtractor(image, samples, opt.img_size, opt.padding, opt.batch_test) # 得到samples的原图
    for i, regions in enumerate(extractor):
        regions = Variable(regions)
        if opt.use_gpu:
            regions = regions.cuda()
        feat = model(regions, out_layer=out_layer)
        if i == 0:
            feats = feat.data.clone()
        else:
            feats = torch.cat((feats, feat.data.clone()), 0)
    return feats


def set_optimizer(model, lr_base, lr_mult=opt.lr_mult, momentum=opt.momentum, w_decay=opt.w_decay):
    '''
    设定优化器
    :param model: 模型
    :param lr_base:  基learning rate
    :param lr_mult:  ??
    :param momentum: 动量
    :param w_decay: 衰变
    :return: 优化器
    '''
    params = model.get_learnable_params() # 获取需要优化的parameter
    param_list = []
    for k, p in params.iteritems(): # 迭代这些参数
        lr = lr_base
        for l, m in lr_mult.iteritems(): #(fc6,10)将fc6层的学习率增加10倍
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr': lr}) # 添加参数和学习率进列表里边
    optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    # TODO:这个param_list（lr）不知道怎么用 我猜是回用到param_list里边的lr， 需要查一下手册
    return optimizer


def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    '''
    在线更新的过程
    :param model: 模型
    :param criterion: 准则
    :param optimizer: 优化器
    :param pos_feats: 正更新样本集 #TODO: 关注一下这个，看看这个就是是feats还是索引
    :param neg_feats: 负更新样本集  #TODO: 验证一下，这个负样本集是非常多的，因为包含hnm提取所需要的样本
    :param maxiter: 更新的最大迭代次数
    :param in_layer: 输入特征的类型
    :return: None
    '''
    model.train() # train模式

    batch_pos = opt.batch_pos
    batch_neg = opt.batch_neg
    batch_test = opt.batch_test
    batch_neg_cand = max(opt.batch_neg_cand, batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0)) # 产生size个数，并打乱
    neg_idx = np.random.permutation(neg_feats.size(0))
    while (len(pos_idx) < batch_pos * maxiter): # 要满足更新需要的样本，32*maxiter
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))]) # 再增加一波样本的索引
    while (len(neg_idx) < batch_neg_cand * maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0 # 两个指针
    neg_pointer = 0

    for iter in range(maxiter): # 训练的主步骤

        # 选择正样本
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long() # 将pos_cur_idx转换为和pos_feats相同类型，并转化为long类型
        pos_pointer = pos_next # 更换到下一个指针

        # 选择负样本
        neg_next = neg_pointer + batch_neg_cand # 每batch_neg_cand个作为一个操作单位
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # 将更新样本打包成mini-batch
        batch_pos_feats = Variable(pos_feats.index_select(0, pos_cur_idx))
        batch_neg_feats = Variable(neg_feats.index_select(0, neg_cur_idx))

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval() # 更新和测试同步进行
            for start in range(0, batch_neg_cand, batch_test): # 每次测试batch_test个
                end = min(start + batch_test, batch_neg_cand)
                score = model(batch_neg_feats[start:end], in_layer=in_layer) # 从in_layer层输入，测试最后的得分
                if start == 0:
                    neg_cand_score = score.data[:, 1].clone() # 保留的是正得分
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.data[:, 1].clone()), 0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats.index_select(0, Variable(top_idx)) # 提取出来的困难负样本
            model.train()

            # 前向过程
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer) # 这个其实不用再前向计算了，因为已经有了

        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), opt.grad_clip)
        optimizer.step()
        # 为了debug方便，这里打印在线更新的过程，最后可以删去
        # print("Iter %d, Loss %.4f" %(iter, loss.data[0]))


def run_mdnet(img_list, init_bbox, gt=None, savefig_dir='', display=False):
    '''
    MDNet测试的主函数
    :param img_list: 跟踪序列
    :param init_bbox: 初始目标框
    :param gt: groundTruth
    :param savefig_dir: 保存的地址
    :param display: 是否显示
    :return: None
    '''

    ###初始化
    ## 初始化目标
    # print('init target')
    target_bbox = np.array(init_bbox) # 给定目标
    result = np.zeros((len(img_list), 4)) # 保存未回归的结果
    result_bb = np.zeros((len(img_list), 4)) # 保存通过BBR回归得到的结果
    result[0] = target_bbox
    result_bb[0] = target_bbox

    ## 初始化模型
    # print('init model')
    model = MDNet(opt.model_path) # 载入模型
    if opt.use_gpu:
        model = model.cuda()
    model.set_learnable_params(opt.ft_layers) # 由前缀来判断，这里是所有fc层

    ## 初始化LOSS和optimizer
    # print('init loss&optimizer')
    criterion = BinaryLoss()
    init_optimizer = set_optimizer(model, opt.lr_init) # 用于初始化的
    update_optimizer = set_optimizer(model, opt.lr_update) # 用于在线更新的


    tic = time.time()  # 保存现在的时间

    ## 载入图片
    # print('load image')
    image = Image.open(img_list[0]).convert('RGB')

    ## 训练BBR模型
    # print('train BBR')
    bbreg_examples = gen_samples(SampleGenerator('uniform', image.size, 0.3, 1.5, 1.1),
                                 target_bbox, opt.n_bbreg, opt.overlap_bbreg, opt.scale_bbreg)
    bbreg_feats = forward_samples(model, image, bbreg_examples)
    bbreg = BBRegressor(image.size)
    bbreg.train(bbreg_feats, bbreg_examples, target_bbox)

    ## 提取用于更新的正负样本
    # 正样本的提取方式为高斯取样，这次为初始帧取
    # print('get_sample')
    pos_examples = gen_samples(SampleGenerator('gaussian', image.size, 0.1, 1.2),
                               target_bbox, opt.n_pos_init, opt.overlap_pos_init)
    # 负样本的提取方式为高斯取样+整体取样，这次为初始帧取
    neg_examples = np.concatenate([
        gen_samples(SampleGenerator('uniform', image.size, 1, 2, 1.1),
                    target_bbox, opt.n_neg_init // 2, opt.overlap_neg_init),
        gen_samples(SampleGenerator('whole', image.size, 0, 1.2, 1.1),
                    target_bbox, opt.n_neg_init // 2, opt.overlap_neg_init)])
    neg_examples = np.random.permutation(neg_examples) # 这里只打乱行

    # 提取正负样本的特征
    # print('extract feature map')
    pos_feats = forward_samples(model, image, pos_examples)
    neg_feats = forward_samples(model, image, neg_examples)
    feat_dim = pos_feats.size(-1) # 这里注意，得到的featuremap格式为：Nx(feat.size())

    ## 初始帧fine-tune
    # print('initialization fine-tune')
    train(model, criterion, init_optimizer, pos_feats, neg_feats, opt.maxiter_init)

    ## 为了后续的提取样本，这里初始化样本提取器 #TODO：上一帧的跟踪结果也可以作为更新时的困难负样本
    sample_generator = SampleGenerator('gaussian', image.size, opt.trans_f, opt.scale_f, valid=True)
    pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2)
    neg_generator = SampleGenerator('uniform', image.size, 1.5, 1.2)

    ## 初始化用作更新的正负样本
    # print('get update-sample')
    pos_feats_all = [pos_feats[:opt.n_pos_update]] # 相当于一个容器
    neg_feats_all = [neg_feats[:opt.n_neg_update]]

    spf_total = time.time() - tic

    ### Display #TODO： 这里需要加强理解
    savefig = savefig_dir != '' # 根据savefig_dir来判断是否保存fig, 这里如果有需要，可以更改到用visdom来显示
    if display or savefig:
        dpi = 80.0
        figsize = (image.size[0] / dpi, image.size[1] / dpi) # 规定fig的尺寸

        fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.]) # Axes相当于一个容器？
        ax.set_axis_off()
        fig.add_axes(ax)
        im = ax.imshow(image, aspect='auto') # auto模式才能显示图片

        if gt is not None:
            gt_rect = plt.Rectangle(tuple(gt[0, :2]), gt[0, 2], gt[0, 3],
                                    linewidth=3, edgecolor="#00ff00", zorder=1, fill=False)
            ax.add_patch(gt_rect)

        rect = plt.Rectangle(tuple(result_bb[0, :2]), result_bb[0, 2], result_bb[0, 3],
                             linewidth=3, edgecolor="#ff0000", zorder=1, fill=False)
        ax.add_patch(rect)

        if display:
            plt.pause(.01)
            plt.draw()
        if savefig:
            fig.savefig(os.path.join(savefig_dir, '0000.jpg'), dpi=dpi)

    ### 主循环
    for i in range(1, len(img_list)): # 注意这里图像的下表也是从0开始：1--第二张图片

        # 记录时间
        tic = time.time()

        # 载入图像
        image = Image.open(img_list[i]).convert('RGB')

        # 获取候选目标框及得分
        samples = gen_samples(sample_generator, target_bbox, opt.n_samples)
        sample_scores = forward_samples(model, image, samples, out_layer='fc6')
        top_scores, top_idx = sample_scores[:, 1].topk(5)  # 提取前5个最高得分，取平均
        top_idx = top_idx.cpu().numpy()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx].mean(axis=0)

        success = target_score > opt.success_thr # 通过阈值来判断是否跟踪成功

        # 如果失败，则扩大搜索范围
        if success:
            sample_generator.set_trans_f(opt.trans_f)
        else:
            sample_generator.set_trans_f(opt.trans_f_expand) #扩大搜索范围

        # BBR回归 #TODO: 级联BBR？？
        if success:
            bbreg_samples = samples[top_idx] # 用最近的跟踪样本来进行回归
            bbreg_feats = forward_samples(model, image, bbreg_samples)
            bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0) # 分别预测，最后也是从5个回归后的框取平均
        else:
            bbreg_bbox = target_bbox # 如果是错误的话，就不进行回归了

        # 如果通过阈值说明跟踪失败，则直接用上一帧的目标
        if not success:
            target_bbox = result[i - 1]
            bbreg_bbox = result_bb[i - 1]

        # 保存结果
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox

        # 收集样本, 只有当跟踪成功时才收集
        if success:
            # 获取更新正负样本
            pos_examples = gen_samples(pos_generator, target_bbox,
                                       opt.n_pos_update,
                                       opt.overlap_pos_update)
            neg_examples = gen_samples(neg_generator, target_bbox,
                                       opt.n_neg_update,
                                       opt.overlap_neg_update)

            # 前向得到feat
            pos_feats = forward_samples(model, image, pos_examples)
            neg_feats = forward_samples(model, image, neg_examples)#TODO： 分组更新这里需要注意(只有成功跟踪才处理)，要修改
            pos_feats_all.append(pos_feats) # 放到容器里, 注意这里容器是List
            neg_feats_all.append(neg_feats)
            if len(pos_feats_all) > opt.n_frames_long: # 抛弃长短时的更新样本
                del pos_feats_all[0]
            if len(neg_feats_all) > opt.n_frames_short:
                del neg_feats_all[0]

        # 短时更新
        if not success:
            nframes = min(opt.n_frames_short, len(pos_feats_all)) # 这里是成功的帧数
            pos_data = torch.stack(pos_feats_all[-nframes:], 0).view(-1, feat_dim) # stack连接成一个新的维度
                                                                                   # (第几帧的样本，Nx（one_feature）)
                                                                               # 然后转成(FxN)x(feat_dim)格式，目的在于打散
                                                                               # 包和帧的样本
            neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)
            # 更新
            train(model, criterion, update_optimizer, pos_data, neg_data, opt.maxiter_update)

        # Long term update
        elif i % opt.long_interval == 0: # 每long_interval帧进行长时更新
            pos_data = torch.stack(pos_feats_all, 0).view(-1, feat_dim)
            neg_data = torch.stack(neg_feats_all, 0).view(-1, feat_dim)
            train(model, criterion, update_optimizer, pos_data, neg_data, opt.maxiter_update)

        spf = time.time() - tic
        spf_total += spf

        # Display
        if display or savefig:
            im.set_data(image) #??

            if gt is not None:
                gt_rect.set_xy(gt[i, :2])
                gt_rect.set_width(gt[i, 2])
                gt_rect.set_height(gt[i, 3])

            rect.set_xy(result_bb[i, :2])
            rect.set_width(result_bb[i, 2])
            rect.set_height(result_bb[i, 3])

            if display:
                plt.pause(.01)
                plt.draw()
            if savefig:
                fig.savefig(os.path.join(savefig_dir, '%04d.jpg' % (i)), dpi=dpi)

        if gt is None:
            print "Frame %d/%d, Score %.3f, Time %.3f" % \
                  (i, len(img_list), target_score, spf)
        else:
            print "Frame %d/%d, Overlap %.3f, Score %.3f, Time %.3f" % \
                  (i, len(img_list), overlap_ratio(gt[i], result_bb[i])[0], target_score, spf)

    # loop外
    fps = len(img_list) / spf_total
    return result, result_bb, fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default='', help='input seq')
    parser.add_argument('-j', '--json', default='', help='input json')
    parser.add_argument('-f', '--savefig', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')

    args = parser.parse_args()
    assert (args.seq != '' or args.json != '')

    # Generate sequence config
    img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)

    # Run tracker
    result, result_bb, fps = run_mdnet(img_list, init_bbox, gt=gt, savefig_dir=savefig_dir, display=display)

    # Save result
    res = {}
    res['res'] = result_bb.round().tolist()
    res['type'] = 'rect'
    res['fps'] = fps
    json.dump(res, open(result_path, 'w'), indent=2)
