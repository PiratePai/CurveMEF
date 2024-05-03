#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:pengp
@file:CurveMEF.py
@time:2023/07/19
"""

import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
from timm.models import register_model


class ds_conv(nn.Module):
    def __init__(self, i, o):
        super(ds_conv, self).__init__()
        self.dw_conv = nn.Conv2d(i, i, kernel_size=3, stride=1, padding=1, bias=True, groups=i)
        self.conv1x1 = nn.Conv2d(i, o, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dw_conv(x)
        out = self.conv1x1(out)
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size = max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.Conv2d(in_size, in_size // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // 4, 1, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Simple_layer(pl.LightningModule):
    def __init__(self, dim_in, dim_out=None, act='relu'):
        super(Simple_layer, self).__init__()
        if dim_out is None:
            dim_out = dim_in
            self.skip = True
        else:
            self.skip = False
        self.conv1 = nn.Sequential(
            ds_conv(dim_in, dim_out),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1, padding=0, bias=True)
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.skip:
            x += identity
        x = self.act(x)
        return x


class Conv1x1_layer(pl.LightningModule):
    def __init__(self, dim_in, dim_out=None, act='relu'):
        super(Conv1x1_layer, self).__init__()
        if dim_out is None:
            dim_out = dim_in
        self.conv = nn.Conv2d(dim_in, dim_out,
                              kernel_size=1, stride=1, bias=True)
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class CheapBootleneck(nn.Module):
    def __init__(
            self, dim_in=6, dim_out=12):
        super(CheapBootleneck, self).__init__()

        self.half = int(dim_in / 2)
        self.dim_out = dim_out
        self.dense1 = Simple_layer(self.half)
        self.dense2 = Simple_layer(self.half)
        self.sim1 = Conv1x1_layer(self.half, self.dim_out)
        # dimout需要是dim_in的1。5倍率
        self.merge = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.dim_out, self.dim_out,
                      kernel_size=1, stride=1, bias=True),
            nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        m = x[:, :self.half]
        m1 = self.dense1(m)
        m2 = self.dense2(m1)
        m3 = torch.cat([m, m1, m2], 1)
        c = x[:, self.half:]
        c = self.sim1(c)
        x = m3 * self.merge(c)
        return x


class Encoder(nn.Module):
    def __init__(self, dim_in=2, dim_mid_1=36, dim_mid_2=48, dim_mid_3=54):
        super(Encoder, self).__init__()
        self.conv_stem = Simple_layer(dim_in, dim_mid_1)
        self.up_1 = CheapBootleneck(dim_mid_1, dim_mid_2)
        self.up_2 = CheapBootleneck(dim_mid_2, dim_mid_3)
        self.dropout = nn.Dropout(0.3)
        self.conv_end = nn.Sequential(
            ds_conv(dim_mid_3, dim_mid_3),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.up_1(x)
        x = self.up_2(x)
        x = self.dropout(x)
        x = self.conv_end(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dim_in=6, dim_mid_1=36, dim_out=3):
        super(Decoder, self).__init__()
        self.channel_mixer = SeModule(dim_in)
        self.conv1 = nn.Sequential(
            ds_conv(dim_in, dim_mid_1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            ds_conv(dim_mid_1, dim_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x + self.channel_mixer(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Estimate(nn.Module):
    def __init__(self, dim_in=6, dim_mid_1=36, dim_mid_2=48, dim_out=3):
        super(Estimate, self).__init__()
        self.down_1 = Simple_layer(dim_in, dim_mid_1, act='relu')
        self.down_2 = Simple_layer(dim_mid_1, dim_mid_2, act='relu')
        self.conv_end = Simple_layer(dim_mid_2, dim_out, act='sigmoid')

    def forward(self, x):
        x = self.down_1(x)
        x = self.down_2(x)
        x = self.conv_end(x)
        return x


class CurveMEFy(pl.LightningModule):
    def __init__(
            self, dim_in=2, dim_out=4):
        super(CurveMEFy, self).__init__()
        # 自编码器主结构
        self.expand = 27
        self.encoder = Encoder(dim_in, 12, 18, self.expand)

        # self.estimate = Estimate(self.expand, dim_out, dim_out)
        self.decoder = Decoder(self.expand, 18, dim_out)
        self.conv_end = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        self.eps = 1e-6

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        # print(1,x.shape)
        x = self.encoder(x)
        x = self.decoder(x)

        # enhance_image = x
        a, b, c1, c2 = x.chunk(4, dim=1)
        nc = c1 + c2 + self.eps
        n1 = c1 / nc
        n2 = c2 / nc
        y1, y2 = enhance_for_train(x1, x2, a, b)
        enhance_image = n1 * y1 + \
                        n2 * y2
        # save_pic(a, './weight_map/a.png', True)
        # save_pic(b, './weight_map/b.png', True)
        # save_pic(c1, './weight_map/c.png', True)
        # save_pic(c2, './weight_map/d.png', True)
        # save_pic(n1*y1, './weight_map/e.png', False)
        # save_pic(n2*y2, './weight_map/f.png', False)
        # save_pic(enhance_image, './weight_map/g.png', False)
        # print(c1)
        # print(c2)
        return enhance_image

    def set_epoch(self, epoch):
        self.epoch = epoch


def sigMoid(x):
    return 1 / (1 + torch.exp(-x))


def mef(a, b, x, y, eps_):
    x = x - a * x * x / (eps_ - x)
    # x = x - a * x * x / (5 - x)

    y = y + b * y * (1 - y) * (1 - y / 2 / eps_)
    # y = y + b * y * (1 - y) * (1 - y / eps_)
    return x, y


def enhance_for_train(x, y, a, b):
    eps_ = 2 / (2 - math.sqrt(2)) + 1e-6
    x, y = mef(a, b, x, y, eps_)
    return x, y


#
# def save_pic(a11, path_temp, flag):
# import torchvision.transforms
# to_pil = torchvision.transforms.ToPILImage()
#     if flag:
#         a11 = a11 * 255
#     else:
#         a11 = a11
#     a_pic = to_pil(a11[0].cpu())
#     a_pic.save(path_temp)
#
# def save_pic2(a11, path_temp):
# import torchvision.transforms
# to_pil = torchvision.transforms.ToPILImage()
#     arr_min_axis21 = a11.min()
#     arr_max_axis31 = a11.max()
#     num1 = a11 - arr_min_axis21  # (100,1,1,1)
#     dem1 = arr_max_axis31 - arr_min_axis21  # (1,1,1,1)
#     pic_ = num1 / dem1  # (100,1,64,64)
#     pic_ = pic_ * 255.0 + (1.0 - pic_) * 0.0
#     a_pic = to_pil(pic_[0].cpu())
#     a_pic.save(path_temp)
def calculate_memory_usage():
    allocated_memory = torch.cuda.memory_allocated() / 1024 ** 2
    reserved_memory = torch.cuda.memory_reserved() / 1024 ** 2
    return allocated_memory, reserved_memory


@register_model
def ghost_curve_2iny(pretrained=False, pth_path=None):
    GhostCurve = CurveMEFy()

    if pretrained:
        state_dict = torch.load(pth_path)
        GhostCurve.load_state_dict(state_dict)
    return GhostCurve


if __name__ == '__main__':
    import torch.cuda as cuda
    import time
    import numpy as np

    tensor_1 = torch.randn(1, 1, 1024, 1024).cuda()
    tensor_2 = torch.randn(1, 1, 1024, 1024).cuda()
    Net = ghost_curve_2iny().eval().cuda()
    cuda.synchronize()
    t1 = time.time()
    torch.autograd.set_detect_anomaly(True)
    with torch.no_grad():
        output = Net(tensor_1, tensor_2)
    cuda.synchronize()
    t2 = time.time()
    allocated_memory, reserved_memory = calculate_memory_usage()
    print(f"Allocated memory: {allocated_memory:.2f} MB")
    print(f"Reserved_memory memory: {reserved_memory:.2f} MB")
    print(t2 - t1)
    parameters = filter(lambda p: p.requires_grad, Net.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
