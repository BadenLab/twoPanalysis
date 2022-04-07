# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 17:25:12 2022

@author: SimenLab
"""
import numpy as np
def get_ops(path):
    ops =  np.load(path, allow_pickle=True)
    ops = ops.item()