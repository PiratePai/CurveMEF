#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:pengp
@file:compute_metric.py
@time:2022/09/17
"""
import time

import cv2
# from PIL import Image, ImageStat
import numpy as np
import pyiqa
import torch
import torch.nn as nn
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, TotalVariation
from torchvision import transforms

loss_psnr = PeakSignalNoiseRatio().cuda()
loss_ssim = MultiScaleStructuralSimilarityIndexMeasure().cuda()
loss_tv = TotalVariation(reduction='none').cuda()
loss_psnr2 = pyiqa.create_metric('psnr')
loss_ssim2 = pyiqa.create_metric('ms_ssim')


def compute_loss(fused, low, high):
    PSNR_loss = loss_psnr(fused, high) + loss_psnr(fused, low)
    SSIM_loss = loss_ssim(fused, high) + loss_ssim(fused, low)
    return PSNR_loss, SSIM_loss


def compute_loss2(fused, low, high):
    loss_tv2 = L_TV()
    PSNR_loss = loss_psnr2(fused, high) + loss_psnr2(fused, low)
    SSIM_loss = loss_ssim2(fused, high) + loss_ssim2(fused, low)
    t1 = time.perf_counter()
    TV_loss = loss_tv2(fused) + loss_tv2(fused)
    t2 = time.perf_counter()
    print(PSNR_loss, SSIM_loss, TV_loss, (t2 - t1) * 1000)
    return PSNR_loss, SSIM_loss, TV_loss


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


def pic2tensor(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    a = transforms.Compose([transforms.ToTensor(), transforms.Resize(256)])
    img = a(img)
    return img.cuda().unsqueeze(0)


def cauclate(pic):
    img = cv2.imread(pic)
    b, g, r = cv2.split(img)
    brightness = np.sqrt(0.241 * (r.mean() ** 2) + 0.691 * (g.mean() ** 2) + 0.068 * (b.mean() ** 2))
    return brightness


if __name__ == '__main__':
    fused_path = "D:\\coding\\new_net\\dataset\\test\\fused\\507_U2PY.tif"
    high_path = 'D:\\coding\\new_net\\dataset\\test\\high\\507.tif'
    low_path = 'D:\\coding\\new_net\\dataset\\test\\low\\507.tif'
    fused_ = pic2tensor(fused_path)
    high_ = pic2tensor(high_path)
    low_ = pic2tensor(low_path)
    compute_loss(fused_, low_, high_)
    compute_loss2(fused_, low_, high_)
    # fused_1 = "D:\\coding\\zero-mef\\workspace\\" \
    #           "shuffle5\\s_48\\output\\AirBellowsGap_U2PY.tif"
    # ev_1 = cauclate(fused_1)
    # print(ev_1)
