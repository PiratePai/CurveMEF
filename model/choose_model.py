#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:pengp
@file:choose_model.py
@time:2022/09/16
"""

from model.CurveMEF import ghost_curve_2in
from model.CurveMEFy import ghost_curve_2iny

model_ = {
    'ghost_curve_2in': ghost_curve_2in,
    'ghost_curve_2iny': ghost_curve_2iny
}


def choose_model(name):
    assert name in model_.keys()
    return model_[name]()
