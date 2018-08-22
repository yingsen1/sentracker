#coding:utf-8

import os
import sys
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

sys.path.insert(0,'/home/wilsontsang/SENTrack')
from sample_generator import *
from utils import *
from train_config import *

class RegionDataset(data.Dataset):
    '''
    训练数据的迭代器
    '''
    def __init__(self, img_dir, img_list, gt):
        '''
        根据所给的目录和文件和GT,给出图片块的数据
        :param img_dir: 图片集的目录
        :param img_list: 某个序列的文件夹的文件名字列表
        :param gt: gt
        :return pos_regions: 正样本的图片块
        :return neg_regions： 负样本的图片块
        '''
        # 初始化内部变量
        self.img_list = np.array([os.path.join(img_dir, img) for img in img_list]) # img_list 保存某个序列所有图片的路径
        self.gt = gt

        self.batch_frames = trainOpt.batch_frames
        self.batch_pos = trainOpt.batch_pos
        self.batch_neg = trainOpt.batch_neg

        self.overlap_pos = trainOpt.overlap_pos
        self.overlap_neg = trainOpt.overlap_neg

        self.crop_size = trainOpt.img_size
        self.padding = trainOpt.padding

        self.index = np.random.permutation(len(self.img_list))  # 随机产生索引
        self.pointer = 0

        image = Image.open(self.img_list[0]).convert('RGB')

        # 正负样本产生器
        self.pos_generator = SampleGenerator('gaussian', image.size, 0.1, 1.2, 1.1, True)
        self.neg_generator = SampleGenerator('uniform', image.size, 1, 1.2, 1.1, True)

    def __iter__(self):
        return self

    def __next__(self):
        # 做一个迭代
        next_pointer = min(self.pointer + self.batch_frames, len(self.img_list)) # min操作防止超出img帧数
        idx = self.index[self.pointer:next_pointer] # 直接取就可以，因为index本来就是散的
        if len(idx) < self.batch_frames: # 当可取的idx少于 batch_frames，则重新置index
            self.index = np.random.permutation(len(self.img_list))
            next_pointer = self.batch_frames - len(idx)  # 这个操作是为了完整地取完一次index
            idx = np.concatenate((idx, self.index[:next_pointer]))
        self.pointer = next_pointer # 类似于循环指针的感觉

        pos_regions = np.empty((0, 3, self.crop_size, self.crop_size))
        neg_regions = np.empty((0, 3, self.crop_size, self.crop_size))
        for i, (img_path, bbox) in enumerate(zip(self.img_list[idx], self.gt[idx])): # zip()组成一个迭代器
            image = Image.open(img_path).convert('RGB')
            image = np.asarray(image) # 将image转换为numpy格式的

            n_pos = (self.batch_pos - len(pos_regions)) // (self.batch_frames - i) # 从每帧取样的样本数
            n_neg = (self.batch_neg - len(neg_regions)) // (self.batch_frames - i) # 每帧采样的数量不一致
            # 取样得到训练样本
            pos_examples = gen_samples(self.pos_generator, bbox, n_pos, overlap_range=self.overlap_pos)
            neg_examples = gen_samples(self.neg_generator, bbox, n_neg, overlap_range=self.overlap_neg)

            pos_regions = np.concatenate((pos_regions, self.extract_regions(image, pos_examples)), axis=0)
            neg_regions = np.concatenate((neg_regions, self.extract_regions(image, neg_examples)), axis=0)

        pos_regions = torch.from_numpy(pos_regions).float() # 转换为FloatTensor格式
        neg_regions = torch.from_numpy(neg_regions).float()
        return pos_regions, neg_regions

    next = __next__


    def extract_regions(self, image, samples):
        '''
        根据坐标和图片获取图片块
        :param image: 图片
        :param samples: 样本的各种坐标
        :return: 图片块
        '''
        regions = np.zeros((len(samples), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(samples):
            regions[i] = crop_image(image, sample, self.crop_size, self.padding, True)

        regions = regions.transpose(0,3,1,2) # Nx(channel,size,size)
        regions = regions.astype('float32') - 128. # 转换为float32类型并中心化
        return regions
