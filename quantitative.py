# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 11:39:06 2022

@author: SimenLab
"""
import pathlib
import numpy as np 
import pandas as pd

import utilities
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
        The n-th trigger by which to average. THis should correspond to how many 
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
    if trigger.ndim == 2:
        trigger = trigger[0]
    if f.dtype != 'float64':
        f = f.astype(float)
    if trigger.dtype != 'float64':
        trigger = trigger.astype(float)
    trig_frames = trigger.nonzero()[0]
    # first_trg_indx = trig_frames[0]
    num_of_trigs = len(np.where(trigger == 1)[0])
    repeats = int(num_of_trigs/mode)
    num_of_frames = f.shape[1]
    
    ## Compute average distance between triggers
    avg_trig_distance = round(np.average(np.gradient(trig_frames)))
    ## Add an "estimated" trigger to the end of trig_frames (such that we also get the last response)
    value_to_add = trig_frames[-1] + avg_trig_distance
    trig_frames = np.append(trig_frames, value_to_add)
    
    print(f"{num_of_trigs} triggers and {repeats} repeats. F-array is {num_of_frames} long")
    
    if 'interpolation_coefficient' in kwargs:    
        interpolation_coefficient = kwargs['interpolation_coefficient']
    else:
        # interpolation_coefficient = 10000
        interpolation_coefficient = num_of_frames * 100 #Upscale by this value
    
    cropped_f, cropped_trig = utilities.data.crop(f, trigger, 0, 0)
    
    # Sometimes the f arrays are misalinged by a single frame, due to triggers
    # occasionally being temporally misaligned. The following algorithm handles
    # this scenario by upsampling the data to a specific temporal resolution
    # (e.g., all arrays are 1000 frames).
    def interpolate_each_trace(f, mode, repeats, interpolation_granularity):
        # Create empty array with the correct shape
        loops_list = np.empty([repeats, f.shape[0], interpolation_granularity])
        for rep in range(repeats):
            
            from_index = trig_frames[(rep)*mode]
            to_index = trig_frames[(rep+1)*mode]
            print("From:", from_index, "to:", to_index, "for rep", rep)
            activity_segment = f[:, from_index:to_index]
            
            # activity_segment = f[:, trig_frames[(
            #     rep-1)*mode]:trig_frames[rep*mode-1]]
            
            interpolated_activitiy_segment = utilities.data.interpolate(activity_segment, output_trace_resolution = interpolation_granularity)
            loops_list[rep] = interpolated_activitiy_segment
        return loops_list
        # return nth_f_loop, nth_trig_loop
    def slice_triggers(c, mode, repeats, interpolation_granularity):
        trig_list = np.empty([round(repeats), interpolation_granularity])
        for rep in range((repeats)):
            # trigger_segment = trigger[trig_frames[(
                # rep-1)*mode]:trig_frames[rep*mode-1]]
            
            from_index = trig_frames[(rep)*mode]
            to_index = trig_frames[(rep+1)*mode]
            trigger_segment = trigger[from_index:to_index]
            # print(trigger_segment)
            # print("Trig segment shape is", trigger_segment.shape)
            interpolated_trig_segment = utilities.data.interpolate(trigger_segment, output_trace_resolution = interpolation_granularity)
            
            trig_list[rep] = interpolated_trig_segment
        return trig_list
    
    # Interpolate and slice as needed 
    trial_traces  = interpolate_each_trace(f, mode, repeats, interpolation_granularity = interpolation_coefficient)
    averaged_traces = np.average(trial_traces, axis = 0)
    trial_triggers = slice_triggers(trigger, mode, repeats, interpolation_granularity = interpolation_coefficient)
    averaged_triggers = np.average(trial_triggers, axis = 0)
    # Interpolate back to oringinal temporal resolution 
    trial_traces = utilities.data.interpolate(trial_traces, int(num_of_frames/repeats)) 
    averaged_traces = utilities.data.interpolate(averaged_traces, int(num_of_frames/repeats)) 
    trial_triggers = utilities.data.interpolate(trial_triggers, int(num_of_frames/repeats))
    averaged_triggers = utilities.data.interpolate(averaged_triggers, int(num_of_frames/repeats)) 
    # Binarise trigger
    trial_triggers = np.where(trial_triggers>0, 1, 0) # Binarise 
    averaged_triggers = np.where(averaged_triggers>0, 1, 0) # Binarise 
    return averaged_traces, trial_traces, trial_triggers, averaged_triggers

