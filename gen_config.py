#coding:utf-8
import os
import json
import numpy as np
import StringIO
from config import *


def gen_config(args):
    '''

    :param args: 输入的一些参数 #TODO： 可以稍微修改修改
    :return:img_list: 跟踪序列的路径
            init_bbox： 初始bbox
            gt      ：该序列的gt
            savefig_dir ：保存fig的地址 (用这个也可以判断是否保存结果)
            args.display ：是否显示图片
            result_path ：存放结果的路径
    '''
    if args.seq != '': # 当给定序列

        # 从给定的序列名中产生跟踪所需要的list、bbox、gt等等
        seq_home = opt.seq_home
        save_home = opt.save_home
        result_home = opt.result_home

        seq_name = args.seq
        img_dir = os.path.join(seq_home, seq_name, 'img') # 序列图片的路径
        gt_path = os.path.join(seq_home, seq_name, 'groundtruth_rect.txt') # 序列gt的路径

        img_list = os.listdir(img_dir) # 返回路径中文件的所有名字
        img_list.sort() # 按顺序进行排列

        ### 有一些序列会出现问题，根据特定的序列进行预处理
        # 返回所有图片绝对路径的列表
        if seq_name == "David":
            img_list = [os.path.join(img_dir, x) for x in img_list[299:770]]
        elif seq_name == "Football1":
            img_list = [os.path.join(img_dir, x) for x in img_list[0:74]]  # actually is 0-73
        elif seq_name == "Freeman3":
            img_list = [os.path.join(img_dir, x) for x in img_list[0:460]]
        elif seq_name == "Freeman4":
            img_list = [os.path.join(img_dir, x) for x in img_list[0:283]]
        elif seq_name == "Diving":
            img_list = [os.path.join(img_dir, x) for x in img_list[0:215]]
        elif seq_name == "Tiger1":
            img_list = [os.path.join(img_dir, x) for x in img_list[5:354]]  # 349 frames is 5-353
        else:
            img_list = [os.path.join(img_dir, x) for x in img_list]

        print(args.seq)

        # 将所有的gt都统一到一种格式上 统一用','来进行分割
        s = open(gt_path).read().replace('\t', ',') # read()为一次读取所有数据，速度比较快
        s = s.replace(' ', ',')
        gt = np.loadtxt(StringIO.StringIO(s), delimiter=',') # 读取txt文件，并根据‘，’分割形成一个列表(普通格式的列表)
        init_bbox = gt[0]

        savefig_dir = os.path.join(save_home, seq_name)
        result_dir = os.path.join(result_home, seq_name)
        if not os.path.exists(result_dir): # 若不存在这个路径，就创建一个 #TODO：这里貌似没有用，需要修改
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir, 'result.json') #TODO: 看看需要这个吗？不需要就删除

    elif args.json != '':
        # load config from a json file

        param = json.load(open(args.json, 'r'))
        seq_name = param['seq_name']
        img_list = param['img_list']
        init_bbox = param['init_bbox']
        savefig_dir = param['savefig_dir']
        result_path = param['result_path']
        gt = None

    if args.savefig:
        if not os.path.exists(savefig_dir): # 保存跟踪得到的fig结果
            os.makedirs(savefig_dir)
    else:
        savefig_dir = ''

    return img_list, init_bbox, gt, savefig_dir, args.display, result_path
