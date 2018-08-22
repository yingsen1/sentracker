#coding:utf-8
from run_tracker import *
# import argparse
from data_prov import *
from config import *

opt._parse()

####单次运行####
class Arg:
    display = False
    seq = ''
    json = ''
    savefig = False

## 运行参数
args = Arg()
args.display = False
args.seq = 'DragonBaby'
args.savefig = False

img_list, init_bbox, gt, savefig_dir, display, result_path = gen_config(args)

## run
result, result_bb, fps = run_mdnet(img_list, init_bbox, gt=gt, savefig_dir=savefig_dir, display=display)
