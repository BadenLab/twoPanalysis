# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 11:39:06 2022

@author: SimenLab
"""
import pathlib
import numpy as np 
import pandas as pd

import utilities
import qualitative
# data_path = r"./Data/data_save/1_8dfp_ntc3_512_1.npz"

def load_data(path):
    load_path = pathlib.Path(path)
    data = np.load(load_path.with_suffix('.npz'), allow_pickle = True)
    return data

def data_struct(npz):
    npz.files
    
# exp_test = load_data(data_path)

def average_signal(f, trigger, mode, **kwargs):
    """
    Parameters
    ----------
    f : Numpy array
        The extracted ROI traces represented as an nxm numpy array
        NB: Remember that Experiment-objects create fs arrays in the format
        [experiment][plane][cell][time], so you likely need at least two indeces
        ([experiment] and [plane]). 
    trigger : Numpy array
        Trigger signal as a 1xn numpy array
    mode : TYPE
        The n-th trigger by which to average. This should correspond to how many 
        repeats/trials of the same stimulus-train there are in a given experiment. 

    **kwargs
    --------
    interpolation_granularity : int 
        Default: 10000 The amount of points to generate after interpolation, independent of 
        what the original input is. Can be specified to any value (but should
        be used carefully...)

    Returns
    -------
    averaged_traces, sliced_traces, sliced_triggers

    """
    
    """
    Take the f trace and from the first trigger, crop out the time interval
    at every 'mode'-interval. Then do this n amount of times until the last 
    trigger, and overlap/average.
    
    """

    # if trigger.ndim == 2:
    #     trigger = trigger[0]
    # if f.dtype != 'float64':
    #     f = f.astype(float)
    # if trigger.dtype != 'float64':
    #     trigger = trigger.astype(float)
    mode = mode - 1
    trig_frames = trigger.nonzero()[0]
    # print(trig_frames)
    # first_trg_indx = trig_frames[0]
    num_of_trigs = len(np.where(trigger == 1)[0])
    repeats = int(num_of_trigs/mode)
    num_of_frames = f.shape[1]
    f_shape = f.shape
    print(f"Mode {mode} with {num_of_trigs} triggers and {repeats} repeats. F-array has shape {f_shape}")
    
    if 'interpolation_coefficient' in kwargs:    
        interpolation_coefficient = kwargs['interpolation_coefficient']
    else:
        interpolation_coefficient = 1000
        # interpolation_coefficient = num_of_frames * 100 #Upscale by this value
    
    # cropped_f, cropped_trig = utilities.data.crop(f, trigger, 0, 0)
    
    # Sometimes the f arrays are misalinged by a few frames, due to triggers
    # occasionally being temporally misaligned. The following algorithm handles
    # this scenario by upsampling the data to a specific temporal resolution
    # (e.g., all arrays are 1000 frames).
    def cut_nth(array, trig_frames, mode, interpolation_granularity):
        ## Trigger occurs in these frames:
        trig_frames
        ## Note of every mode-th trigger to cut from the given array 
        cut_points = trig_frames[::mode] 
        ## Make a list for array segments to be placed in 
        segment_list = []
        ## Segment_list has to be 1 dim more than input array.ndim where dim 0 is always time/interpolation_coefficient
        ## such that repeats (e.g. segments we are cutting out) can be placed in the new dimension
        _shape = list(np.expand_dims(array, 0).shape)
        # _shape[0] = repeats
        _shape[0] = len(cut_points)-1
        _shape[-1] = interpolation_granularity
        segment_list = np.empty((_shape))
        # print(_shape)
        # print(array.ndim)
        # Loop through list of points to cut from
        # print(cut_points)
        # print("cutpoints is this long:",len(cut_points))
        for i in range(len((cut_points))-1):
            print(i)
            ## Make indeces to cut from and to
            from_index = cut_points[i]
            to_index = cut_points[i+1]
            # print(f"For iteration {i} cut from {from_index} to {to_index}")
            ## Slice indeces from input array
            if array.ndim == 1:
                array_segment = array[from_index:to_index]
            if array.ndim == 2:
                array_segment = array[:, from_index:to_index]
            
            # if array.ndim == 3:
                # array_segment = array[:, from_index:to_index] 
            ## Interpolate the segment to standardise length (in case of trigger missalignment)
            interpolated_segmented = utilities.data.interpolate(array_segment, interpolation_coefficient)
            # print(np.max(interpolated_segmented), np.min(interpolated_segmented))
            ## Append the segments to the segment list
            segment_list[i] = interpolated_segmented
            # print(np.max(segment_list[i]), np.min(segment_list[i]))
            # segment_list.append(interpolated_segmented)
            ## Insert segments to segment array
            # for i in interpolated_segmented:
                # np.insert(segment_list, 1, interpolated_segmented, axis = 0)
            # print(i)
        print(np.max(segment_list[i]), np.min(segment_list[i]))    
        return segment_list
        

    ## Interpolate and slice as needed 
    # print(f.shape)
    trial_traces  = cut_nth(f, trig_frames, mode, interpolation_granularity = interpolation_coefficient)
    print(np.max(trial_traces), np.min(trial_traces))
    trial_triggers = cut_nth(trigger, trig_frames, mode, interpolation_granularity = interpolation_coefficient)
    averaged_traces = np.average(trial_traces, axis = 0)
    ## Interpolate back to oringinal temporal resolution 
    trial_triggers = utilities.data.interpolate(trial_triggers, int(num_of_frames/mode))
    trial_traces = utilities.data.interpolate(trial_traces, int(num_of_frames/mode))
    averaged_traces = utilities.data.interpolate(averaged_traces, int(num_of_frames/mode)) 
    # # ## Binarise trigger
    trial_triggers = np.where(trial_triggers>0, 1, 0) # Binarise 
    
    return averaged_traces, trial_traces, trial_triggers
    # return trial_traces

## Testing
# # trials = average_signal(fz, trigz, 30)
averaged_traces, trial_traces, trial_triggers = average_signal(fz, trigz, 10)
experiment = 0
roi = 5
qualitative.plot_averages(averaged_traces, trial_traces, trial_triggers, roi)
