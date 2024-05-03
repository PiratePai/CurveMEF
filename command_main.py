#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
@author:pengp
@file:command_main.py
@time:2023/11/24
"""
import os

if __name__ == '__main__':
    # dir_list is the folder name of config files
    dir_list = ['curve']
    for dir_ in dir_list:
        path_dir = 'config/{}'.format(dir_)
        # item_list is the suffix name of config files in that folder
        item_list = ['rgb', 'y']
        for item_ in item_list:
            cfg_path = 'config/{}/curvemef_{}.yml'.format(dir_, item_)
            os.system('python train.py -config {}'.format(cfg_path))
