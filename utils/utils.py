#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:pengp
@file:utils.py
@time:2022/04/22
"""
from copy import deepcopy

import pytorch_lightning.callbacks as plc
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme


def build_from_cfg(cfg_input, args_):
    arg_dict = vars(args_)
    cfg_temp = deepcopy(cfg_input)
    for key, value in cfg_temp.items():
        arg_dict[key] = value
    return args_


progress_bar = plc.RichProgressBar(
    theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="grey82",
    )
)


def load_callbacks():
    callbacks = [
        ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints/',
            filename='model-{epoch:02d}-{val_loss:.2f}',
            save_top_k=3,
            mode='min',
        ),
        TQDMProgressBar(refresh_rate=1)
    ]

    return callbacks
