#coding:utf-8
import sys
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

from utils import *

'''
CLASS RegionExtractor
'''

class RegionExtractor():
    '''
       根据坐标切割图像块，形成batch
       初始化
           :param image: 图像
           :param samples:  samples的坐标
           :param crop_size: 切割的尺寸
           :param padding: 填充的参数
           :param batch_size: batch_size
           :param shuffle: 是否打乱，默认不打乱

       :return regions 预处理并且打包形成的regions(tensor) Nx(3,size,size)
       '''
    def __init__(self, image, samples, crop_size, padding, batch_size, shuffle=False):

        self.image = np.asarray(image) # 转化为ndarray
        self.samples = samples # Nx(x_min, y_min, w, h)
        self.crop_size = crop_size
        self.padding = padding
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.index = np.arange(len(samples)) # samples每行的索引
        self.pointer = 0

        self.mean = self.image.mean(0).mean(0).astype('float32') # 图像所有像素的平均

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.samples): # 当迭代完所有得samples
            self.pointer = 0
            raise StopIteration
        else:
            next_pointer = min(self.pointer + self.batch_size, len(self.samples)) # 设定下一次迭代的指针
            index = self.index[self.pointer:next_pointer] # 一个batch的所有指针
            self.pointer = next_pointer # 转到下一次的迭代指针

            regions = self.extract_regions(index)
            regions = torch.from_numpy(regions) # 转化为tensor
            return regions
    next = __next__

    def extract_regions(self, index):
        '''
        根据batch包含的索引，来分别截取image_patch
        :param index: 索引，一位数组
        :return: N x (3, size, size)的image_patch
        '''
        regions = np.zeros((len(index),self.crop_size,self.crop_size,3),dtype='uint8') # N x (size,size,3)
        for i, sample in enumerate(self.samples[index]): #根据index分别索引
            regions[i] = crop_image(self.image, sample, self.crop_size, self.padding)

        # 预处理
        regions = regions.transpose(0,3,1,2).astype('float32') # N x (3, size, size)
        regions = regions - 128 # 去均值， 中心化
        return regions
