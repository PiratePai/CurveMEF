#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:pengp
@file:__init__.py
@time:2022/04/22
"""

from .choose_model import choose_model
from .model_interface import MInterface

__all__ = [
    "MInterface",
    'choose_model'
]
