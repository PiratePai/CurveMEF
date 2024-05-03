#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:pengp
@file:main.py
@time:2022/02/27
"""
import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch

from data import load_config, cfg, DInterface
from model import MInterface
from utils import mkdir, FastMEFLogger, \
    load_callbacks, build_from_cfg
from lightning.pytorch.loggers import WandbLogger

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


def main(args):
    wandb_logger = WandbLogger(project=args.pro_name, log_model='all', save_dir=args.save_dir, name=args.config)
    # torch.autograd.set_detect_anomaly(True)

    load_config(cfg, args.config)
    local_rank = int(args.local_rank)
    mkdir(local_rank, cfg.save_dir)
    mkdir(local_rank, Path(cfg.save_dir).joinpath('best'))
    mkdir(local_rank, Path(cfg.save_dir).joinpath('output'))
    mkdir(local_rank, Path(cfg.save_dir).joinpath('samples'))
    l1 = FastMEFLogger(cfg.save_dir)
    l1.dump_cfg(cfg)
    logger = [l1, wandb_logger]
    #
    if args.seed is not None:
        l1.info("Set random seed to {}".format(args.seed))
        pl.seed_everything(args.seed)
    args_ = build_from_cfg(cfg, args)
    l1.info("Setting up data...")
    data_module = DInterface(args_.datas)
    l1.info('Ctrateing model...')
    model = MInterface(args_)
    # 载入模型
    model_resume_path = (
        Path(cfg.save_dir).joinpath('model_last.ckpt')
        if "resume" in cfg
        else None
    )
    args_.callbacks = load_callbacks()
    # 梯度裁剪 计算
    weight_list = list(args_.loss_term.weights)
    x_limit = 0
    for weight_item in weight_list:
        x_limit += args_.loss_term.weights[weight_item]
    trainer = pl.Trainer(
        default_root_dir=cfg.save_dir,
        max_epochs=cfg.total_epochs,
        devices="auto", accelerator="auto",
        logger=logger,
        gradient_clip_val=int(x_limit),
        accumulate_grad_batches=1,
        profiler="simple",
        check_val_every_n_epoch=cfg.val_intervals.num,
        log_every_n_steps=cfg.log_interval,
        resume_from_checkpoint=model_resume_path,
        precision=cfg.precision,
        callbacks=load_callbacks(),
    )
    trainer.fit(model=model, datamodule=data_module)


def make_sample(args):
    if args.seed is not None:
        pl.seed_everything(args.seed)
    args_ = build_from_cfg(cfg, args)
    model = MInterface(args_)

    path_1 = args_.save_dir
    dir_list = Path(path_1).joinpath('best').joinpath('model_best').glob('*')
    f_str = 'model_best_epoch_'
    pic_path = Path(path_1).joinpath('samples')
    from torchvision import transforms
    from PIL import ImageDraw, ImageFont
    test_tensor = torch.load('test1pngy.pt')
    to_pil = transforms.ToPILImage()
    for test_weight in dir_list:
        if f_str in str(test_weight):
            epoch_id = Path(test_weight).stem.split('_')[-1]
            model = model.load_from_checkpoint(test_weight)
            model.eval().cuda()
            x_, y_ = test_tensor
            text_ = 'Epoch_{}'.format(epoch_id)
            with torch.no_grad():
                f_ = model.forward(x_, y_)
            f_ = f_.squeeze(0).cpu()
            f_ = to_pil(f_)
            draw = ImageDraw.Draw(f_)
            ttf_path = "simhei.ttf"
            font = ImageFont.truetype(r'{}'.format(ttf_path), size=50)
            draw.text((50, 50), text_, fill=(255, 0, 0), font=font)
            f_.save(pic_path.joinpath("Epoch_{}.png".format(epoch_id)))


def make_movie(args):
    if args.seed is not None:
        pl.seed_everything(args.seed)
    args_ = build_from_cfg(cfg, args)
    path_1 = args_.save_dir
    pic_path = Path(path_1).joinpath('samples')
    import cv2
    import os
    images = [img for img in os.listdir(pic_path) if img.endswith(".png")]
    # sort
    images.sort(key=lambda x: int(Path(x).stem.split("_")[1]))
    frame = cv2.imread(os.path.join(pic_path, images[0]))
    height, width, layers = frame.shape
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
    # create video
    video = cv2.VideoWriter(os.path.join(pic_path, 'Epoch_Final.mp4'), fourcc, 24, (width, height))

    # write pics into video
    for image in images:
        # read picture
        img = cv2.imread(os.path.join(pic_path, image))
        # write pic in 12 frame/s
        for i in range(12):
            video.write(img)
    # release the object
    video.release()


def train(args_):
    import time
    import torch.cuda as cuda
    load_config(cfg, args_.config)
    args__ = build_from_cfg(cfg, args_)
    cuda.synchronize()
    t1 = time.time()
    # train model
    main(args__)
    cuda.synchronize()
    t2 = time.time()
    print('训练共耗时{}ms'.format((t2 - t1) * 1000))
    # # create sample of each epoch
    # make_sample(args__)
    # # transform samples into video
    # make_movie(args__)


if __name__ == '__main__':
    args1 = parse_args()
    train(args1)
