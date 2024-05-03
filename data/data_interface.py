#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:pengp
@file:data_interface.py
@time:2022/04/17
@reference: miracleyoo/pytorch-lightning-template
"""
from pathlib import Path

import pytorch_lightning as pl
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from data.data_utils import read_path_txt, read_pic


class DInterface(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.batch_size = cfg.batch_size
        self.train_path = cfg.train_path
        self.test_path = cfg.test_path
        self.num_workers = cfg.num_workers
        self.channel = cfg.channel

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            train_set = HLDataset(self.train_path, self.channel, stage)
            train_set_size = int(len(train_set) * 0.8)
            valid_set_size = len(train_set) - train_set_size
            self.trainset, self.valset = data.random_split(train_set, [train_set_size, valid_set_size])
        if stage == 'test' or stage is None:
            self.testset = HLDataset(self.test_path, self.channel, stage)

    def train_dataloader(self):
        return DataLoader(self.trainset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.testset,
                          batch_size=1,
                          num_workers=self.num_workers,
                          shuffle=False)


class HLDataset(Dataset):

    def __init__(self, path_, channel=1, stage=None):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        self.path = path_
        self.high_dir = read_path_txt(Path(self.path).joinpath('high'))
        self.low_dir = read_path_txt(Path(self.path).joinpath('low'))
        self.channel = channel
        self.stage = stage

    def __getitem__(self, idx):
        high_img_, high_img_stem_x, suffix_x = read_pic(self.high_dir[idx], self.channel)
        low_img_, stem_x_, suffix_x_ = read_pic(self.low_dir[idx], self.channel)
        c1, h1, w1 = high_img_.shape
        c2, h2, w2 = low_img_.shape
        if h1 != h2 and h1 == w2:
            high_img_ = high_img_.permute(0, 2, 1)
        if self.stage == 'fit':
            return high_img_, low_img_
        else:
            return high_img_, low_img_, self.high_dir[idx], self.low_dir[idx]

    def __len__(self):
        assert len(self.high_dir) == len(self.low_dir)
        return len(self.high_dir)


if __name__ == "__main__":
    path_1 = '/home/pp/ALgorithms/git_curvemef/dataset/data9/train'
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.RandomCrop(256),
         transforms.Normalize(mean=[0.5], std=[0.5])])
    E = HLDataset(path_1, channel=1, stage='test')
    print(E.__getitem__(0)[3])
