#coding:utf-8
from pprint import pprint # 格式化输出

class TrainConfig():
    # 用GPU训练
    use_gpu = True

    # 训练集路径
    seq_home = '../../tracking_benchmark/dataset/' # 图片集的路径
    seqlist_path = 'data/vot-otb.txt' # 图片序列的路径
    output_path = 'data/vot-otb.pkl' # 输出模型的路径 #TODO：确认一下

    # 模型路径
    init_moedel_path = '../models/imagenet-vgg-m.mat' # imagenet预训练模型
    model_path = '../models/mdnet_vot-otb_seblock_0825_02.pth'

    # mini-batch的设置
    batch_frames = 8 # 每个batch只取8帧
    batch_pos = 32
    batch_neg = 96

    # overlap的设置
    overlap_pos = [0.7, 1]
    overlap_neg = [0, 0.5]

    # 裁切图片的设置
    img_size = 107
    padding = 16

    # 优化器的设置
    lr = 1e-4
    w_decay = 5e-4
    momentum = 0.9
    grad_clip = 10
    ft_layers = ['conv', 'fc'] #TODO：注意这里
    lr_mult = {'fc': 10}
    n_cycles = 100 # 50


    def _parse(self):
        state_dict = self._state_dict()

        # 设置完毕，打印
        print('======================user-config========================')
        pprint(self._state_dict())
        print('=========================end=============================')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in TrainConfig.__dict__.items() \
                if not k.startswith('_')}

trainOpt = TrainConfig()


