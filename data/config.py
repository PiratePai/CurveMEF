#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:pengp
@file:config.py
@time:2022/02/27
"""
import yaml

from utils.yacs import CfgNode

cfg = CfgNode(new_allowed=True)
cfg.save_dir = "../util/"
# common params for NETWORK
cfg.load_model = ''
# logger
cfg.log_interval = 50
# testing
# # data need
cfg.datas = CfgNode(new_allowed=True)
cfg.datas.train_path = 'dataset/data1/train'
cfg.datas.test_path = 'dataset/data1/test'
cfg.net_struct = CfgNode(new_allowed=True)
cfg.optim_term = CfgNode(new_allowed=True)
cfg.optim_term.loss_term = CfgNode(new_allowed=True)
cfg.optim_term.optimizer = CfgNode(new_allowed=True)
cfg.optim_term.warmup = CfgNode(new_allowed=True)
cfg.optim_term.lr_schedule = CfgNode(new_allowed=True)


def load_config(cfg, args_cfg):
    cfg.defrost()
    cfg.merge_from_file(args_cfg)
    cfg.freeze()


if __name__ == "__main__":
    cfg_1 = load_config(cfg, 'D:\\coding\\zero-mef\\config\\fast\\s_1.yml')
    with open('1.yaml', mode='w', encoding='utf-8') as f:
        yaml.dump(cfg, f)
    if cfg.loss_term.struct_loss:
        print(1)
    else:
        print(0)
