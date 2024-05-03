#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:pengp
@file:data_utils.py
@time:2022/05/04
"""
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def read_path_txt(path):
    f = []  # image files
    for p in path if isinstance(path, list) else [path]:
        p = str(Path(p))  # os-agnostic 无关系统的目录操作
        parent = str(Path(p).parent) + os.sep  # 获取父目录
        if Path(p).is_file():  # file
            with open(p, 'r') as t:
                t = t.read().splitlines()
                f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
        elif Path(p).is_dir():  # folder
            f += glob.iglob(p + os.sep + '*.*')
        else:
            raise Exception('%s does not exist' % p)
    return f


def pic2tensor(path):
    img = Image.open(path)
    img = (np.asarray(img) / 255.0)
    return img


def y2tensor(path):
    img = cv2.imread(path, 0)
    img = np.expand_dims(img, axis=2) / 255.0
    return img


def read_pic(pic_path, channel=1):
    if channel == 1:
        img_tensor = y2tensor(pic_path)
    else:
        img_tensor = pic2tensor(pic_path)
    img = torch.from_numpy(img_tensor).float()
    pic_x = img.permute(2, 0, 1)
    stem_x, suffix_x = Path(pic_path).stem, Path(pic_path).suffix
    return pic_x, stem_x, suffix_x


def tensor2save(fused_pic, batch, save_path, channel):
    fused_pic = fused_pic.cpu()
    [h_num, h_ch, h_h, h_w] = fused_pic.shape
    high_path, low_path = batch
    for idx in range(h_num):
        pic_ = fused_pic[idx]
        path_ = high_path[idx]
        path_temp = str(Path(save_path).joinpath(Path(path_).stem + Path(path_).suffix))
        if channel == 1:
            arr_min_axis21 = pic_.min()
            arr_max_axis31 = pic_.max()
            num1 = pic_ - arr_min_axis21  # (100,1,1,1)
            dem1 = arr_max_axis31 - arr_min_axis21  # (1,1,1,1)
            pic_ = num1 / dem1  # (100,1,64,64)
            pic_ = pic_ * 255.0 + (1.0 - pic_) * 0.0
            pic_ = pic_.data.squeeze().cpu().numpy()
            cv2.imwrite(path_temp, pic_)
        else:
            torchvision.utils.save_image(pic_, path_temp)


if __name__ == '__main__':
    transforms_1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(256)])
    pic_1 = "D:\\coding\\dataset\\P2\\1\\1.JPG"
    pic_2 = 'D:\\coding\\CNN-Trans\\dataset\\data9\\train\\gt\\a0001_rd_0.jpg'
    A = read_pic(pic_2, 3)
