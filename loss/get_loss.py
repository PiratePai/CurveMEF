#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:pengp
@file:get_loss.py
@time:2022/09/16
"""

import torch
import torch.nn as nn
from torchmetrics import StructuralSimilarityIndexMeasure


class L_exp(nn.Module):
    def __init__(self, device):
        super(L_exp, self).__init__()
        # print(1)
        self.device = device
        self.pool = nn.AvgPool2d(8)
        self.mean_val = 0.43

    def forward(self, y, x):
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        torch_val = torch.FloatTensor([self.mean_val]).to(self.device)
        d = torch.mean(torch.pow(mean - torch_val, 2))
        return d


class SSIM(StructuralSimilarityIndexMeasure):
    def __init__(self, data_range=1.0, **kwargs):
        super(SSIM, self).__init__(data_range=data_range, **kwargs)

    def forward(self, pred, target):
        return 1.0 - super(SSIM, self).forward(pred, target)


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, pred, target):
        return torch.mean(self.mse_loss(pred, target))


mse = MSE()
loss_ssim = SSIM(data_range=1.0)


def get_loss(cfg_, device='cuda'):
    loss_dict = {'L1': nn.L1Loss(), 'MSE': mse.to(device), 'SSIM': loss_ssim.to(device),
                 'LEX': L_exp(device)}
    loss_name_list = list(cfg_.name)
    return_model_list = []
    return_weight_list = []
    for item in loss_name_list:
        loss_name = cfg_.name[item].upper()
        loss_model = loss_dict[loss_name]
        return_model_list.append(loss_model)
        loss_weights = cfg_.weights[item]
        return_weight_list.append(loss_weights)
    return loss_name_list, return_model_list, return_weight_list


def compute_loss(model_list, weight_list, gt, fused):
    value_list = [model(gt, fused) for model in model_list]
    loss_list = [weight * value for weight, value in zip(weight_list, value_list)]  # 乘以权重
    return value_list, loss_list