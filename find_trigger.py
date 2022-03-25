# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 15:20:34 2022

@author: SimenLab
"""
import numpy as np 

def find_trigger(trgr_arr):
   frame_log = np.empty(trgr_arr.shape[0])
   for frame in range(trgr_arr.shape[0]):
       if np.any(trgr_arr[frame] > 5000):
           frame_log[frame] = 1
       else:
           frame_log[frame] = 0
   return frame_log