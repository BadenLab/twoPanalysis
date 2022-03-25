# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 16:43:42 2021

@author: SimenLab
"""

import os
def change_dir():
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    print(dname)
    os.chdir(dname)