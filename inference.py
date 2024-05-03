#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:pengp
@file:inference.py
@time:2024/04/26
"""
import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch

from data import load_config, cfg, DInterface
from model import MInterface
from utils import load_callbacks, build_from_cfg

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

cdg_name = 'curvemef_rgb'


def parse_args():
    parser = argparse.ArgumentParser()  # 创建一个解析器
    parser.add_argument(
        "-config", default='config/curve/{}.yml'.format(cdg_name), help="train config file path")
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="node rank for distributed training")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()
    return args

def infer(args):
    if args.seed is not None:
        pl.seed_everything(args.seed)
    #
    args_ = build_from_cfg(cfg, args)
    # 梯度裁剪 计算
    weight_list = list(args_.loss_term.weights)
    x_limit = 0
    for weight_item in weight_list:
        x_limit += args_.loss_term.weights[weight_item]
    # print(x_limit)
    data_module = DInterface(args_.datas)
    model = MInterface(args_)
    # 载入模型
    trainer = pl.Trainer(
        default_root_dir=cfg.save_dir,
        max_epochs=cfg.total_epochs,
        devices="auto", accelerator="auto",
        gradient_clip_val=int(x_limit),
        accumulate_grad_batches=1,
        profiler="simple",
        check_val_every_n_epoch=cfg.val_intervals.num,
        log_every_n_steps=cfg.log_interval,
        callbacks=load_callbacks(),
        precision=cfg.precision,
    )

    test_weights = Path(args_.save_dir).joinpath('best/model_best/model_best.ckpt')
    model = model.load_from_checkpoint(test_weights)
    trainer.test(model=model, datamodule=data_module)


def inference(args_):
    import time
    import torch.cuda as cuda
    load_config(cfg, args_.config)
    args__ = build_from_cfg(cfg, args_)
    cuda.synchronize()
    t3 = time.time()
    infer(args__)
    cuda.synchronize()
    t4 = time.time()
    print('推理共耗时{}ms'.format((t4 - t3) * 1000))
    # # create sample of each epoch
    # make_sample(args__)
    # # transform samples into video
    # make_movie(args__)


if __name__ == '__main__':
    args1 = parse_args()
    inference(args1)
