# -*- coding: utf-8 -*-
"""
Created on Fri May 13 16:43:54 2022

@author: SimenLab
"""

import utilities
import numpy as np

f = np.random.rand(14, 1229)
trig_frames = np.arange(12, 1003, 30)
mode = 5
trig = np.random.randint(2, size=(1, 1229))

def cut_n_interp(array, trig_frames, mode, interpolation_granularity):
    ## Trigger occurs in these frames:
    trig_frames
    ## Make a list for array segments to be placed in 
    segment_list = []
    ## Note of every mode-th trigger to cut from the given array 
    cut_points = trig_frames[::mode] 
    ## Loop through list of points to cut from
    for i in range(len((cut_points))-1):
        ## Make indeces to cut from and to
        from_index = cut_points[i]
        to_index = cut_points[i+1]
        # print(f"For iteration {i} cut from {from_index} to {to_index}")
        ## Slice indeces from input array
        array_segment = array[:, from_index:to_index]
        print(array_segment)
        ## Append these slices to the segment list
        segment_list.append(array_segment)
    return segment_list

a = cut_n_interp(f, trig_frames, mode, 100)
b = cut_n_interp(trig, trig_frames, mode, 100)