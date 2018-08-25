#coding:utf-8
from pprint import pprint # 打印所有数据

'''
关于算法中超参的设置
'''
class Config():

    # 是否显示跟踪图片
    #display = False
    # 是否保存fig跟踪结果
    #savefig = False

    # use GPU or CPU
    use_gpu = True

    # about path #添加其他保存的模型的路径
    model_path = 'models/mdnet_vot-otb_seblock_0825_01.pth'
    seq_home = '/home/wilsontsang/tracking_benchmark/dataset/OTB'
    save_home = './result_fig' # 保存跟踪结果的路径
    result_home = './result' # 保存跟踪得到的每帧坐标

    # about im_crop
    img_size = 107
    padding = 16

    # about batch
    batch_pos = 32
    batch_neg = 96
    batch_neg_cand = 1024 # 提取困难负样本时候选的负样本
    batch_test = 256

    # about sampleGenerator
    n_samples = 256 # 候选目标框的数量
    trans_f = 0.6
    scale_f = 1.05
    trans_f_expand = 1.5

    # about bbox regression model
    n_bbreg = 1000
    overlap_bbreg = [0.6, 1]
    scale_bbreg = [1, 2]

    # about init
    lr_init = 1e-4
    maxiter_init = 30 # 测试初始化迭代30次
    n_pos_init = 500 # 样本数量
    n_neg_init = 5000
    overlap_pos_init = [0.7, 1]
    overlap_neg_init = [0, 0.5]

    # about online-update
    lr_update = 2e-4
    maxiter_update = 15 # 在线更新时，只迭代15次
    n_pos_update = 50
    n_neg_update = 200
    overlap_pos_update = [0.7, 1]
    overlap_neg_update = [0, 0.3]

    # 长短时更新
    n_frames_short = 20
    n_frames_long = 100
    long_interval = 10 # 长时更新的间隔时间

    # 成功跟踪的标志(阈值)
    success_thr = 0

    # 优化算法
    w_decay = 5e-4
    momentum = 0.9
    grad_clip = 10
    lr_mult = {'fc6': 10}
    ft_layers = ['fc'] # 需要finetune的层

    # group-update
    num_per_group = 5
    group_choice = 1

    # 需要接受kwargs
    # def _parse(self, kwargs):
    #     state_dict = self._state_dict()
    #     for k, v in kwargs.items():  # 接受kwargs的key,value，并更改option
    #         if k not in state_dict:
    #             raise ValueError('Unknown Option: --%s' %(k))
    #         setattr(self, k, v) # 设置
    #
    #     # 设置完毕，打印
    #     print('=====user config=====')
    #     pprint(self._state_dict())
    #     print('=========end=========')

    def _parse(self):
        state_dict = self._state_dict()

        # 设置完毕，打印
        print('=====================user config=========================')
        pprint(self._state_dict())
        print('=========================end=============================')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

opt = Config()