#coding:utf-8
import os
import numpy as np
import pickle
from collections import OrderedDict

from train_config import *

'''
将训练用的数据保存成data字典中，并输出到output_path/xxx.plk
'''

seq_home = trainOpt.seq_home
seqlist_path = trainOpt.seqlist_path
output_path = trainOpt.output_path

# 打开文件，读取训练集的列表
with open(seqlist_path, 'r') as fp:
    seq_list = fp.read().splitlines() # 整体读取，并根据行组成列表

data = {}
for i, seq in enumerate(seq_list):
    print(seq)
    # 针对某一个序列，如果为.jpg文件，则加入并进行排序
    img_list = sorted([p for p in os.listdir(seq_home+seq) if os.path.splitext(p)[1] == '.jpg'])
    if seq == 'vot2014/ball':
        img_list.pop() # 这个序列有点问题，GT只有602个，但是有603张照片

    gt = np.loadtxt(seq_home+seq+'/groundtruth.txt', delimiter=',') # 可能需要更改一下分裂的标准,gt为一个列表
    print(len(img_list)) # 打印下序列图像的个数
    print(len(gt)) # 打印下gt的个数 # 默认是行的个数
    assert len(img_list) == len(gt), 'Lengths do not match!!'

    # 将gt转换为特定的格式(x_min,y_min,w,h)
    if gt.shape[1] == 8: # 为什么是8个？
        x_min = np.min(gt[:, [0,2,4,6]], axis=1)[:, None] # None的作用是将一行列表转换为以这个为列，增加一个行的维度
        y_min = np.min(gt[:, [1,3,5,7]], axis=1)[:, None]
        x_max = np.max(gt[:, [0,2,4,6]], axis=1)[:, None]
        y_max = np.max(gt[:, [1,3,5,7]], axis=1)[:, None]
        gt = np.concatenate((x_min, y_min, x_max-x_min, y_max-y_min),axis=1)

    data[seq] = {'images':img_list, 'gt':gt} # 字典中的字典

with open(output_path, 'wb') as fp: # wp：以二进制格式打开一个文件只用于写入。如果该文件已存在则打开文件，并从开头开始编辑，
    pickle.dump(data, fp, -1)
