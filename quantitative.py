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
            activity_segment = f[:, trig_frames[(
                rep-1)*mode]:trig_frames[rep*mode-1]]
            interpolated_activitiy_segment = utilities.data.interpolate(activity_segment, output_trace_resolution = interpolation_granularity)
            loops_list[rep] = interpolated_activitiy_segment
        return loops_list
        # return nth_f_loop, nth_trig_loop
    def slice_triggers(trigger, interpolation_granularity):
        trig_list = np.empty([round(repeats), interpolation_granularity])
        for rep in range(round(repeats)):
            trigger_segment = trigger[trig_frames[(
                rep-1)*mode]:trig_frames[rep*mode-1]]
            interpolated_trig_segment = utilities.data.interpolate(trigger_segment, output_trace_resolution = interpolation_granularity)
            trig_list[rep] = interpolated_trig_segment
        return trig_list
    
    # Interpolate and slice as needed 
    trial_traces  = interpolate_each_trace(f, mode, repeats, interpolation_granularity = interpolation_coefficient)
    trial_triggers = slice_triggers(trigger, interpolation_granularity = interpolation_coefficient)
    averaged_traces = np.average(trial_traces, axis = 0)
    # Interpolate back to oringinal temporal resolution 
    trial_triggers = utilities.data.interpolate(trial_triggers, int(num_of_frames/repeats))
    # print(trial_traces.shape)
    trial_traces = utilities.data.interpolate(trial_traces, int(num_of_frames/repeats)) 
    averaged_traces = utilities.data.interpolate(averaged_traces, int(num_of_frames/repeats)) 
    # Binarise trigger
    trial_triggers = np.where(trial_triggers>0, 1, 0) # Binarise 
    return averaged_traces, trial_traces, trial_triggers


#Load data
# def load_datav1(path):
#     load_path = pathlib.Path(path)    
#     with np.load(load_path.with_suffix('.npz'), allow_pickle = True) as data:
#         f_cells     = data["f_cells"]
#         f_neuropils = data["f_neuropils"] 
#         spks        = data["spks"]
#         stats_file  = data["stats_file"] 
#         iscell      = data["iscell"]
#         stats       = data["stats"] 
#         ops         = data["ops"]
#         db          = data["db"] 
#         output_ops  = data["output_ops"] 
#         trigger_arr = data["trigger_arr"]
#         header_info = data["header_info"]
        
#         return f_cells
    
#         "This works kinda, but dicts like ops are just empty..."
        
# def load_datav2(path):
#     load_path = pathlib.Path(path)    
#     with np.load(load_path.with_suffix('.npz'), allow_pickle = True) as data:
#         return data
#         # f_cells     = data["f_cells"]
#         # f_neuropils = data["f_neuropils"] 
#         # spks        = data["spks"]
#         # stats_file  = data["stats_file"] 
#         # iscell      = data["iscell"]
#         # stats       = data["stats"] 
#         # ops         = data["ops"]
#         # db          = data["db"] 
#         # output_ops  = data["output_ops"] 
#         # trigger_arr = data["trigger_arr"]
#         # header_info = data["header_info"]
        

#         # data_dict = {
#         #     "f_cells"     : f_cells,
#         #     "f_neuropils" : f_neuropils, 
#         #     "spks"        : spks, 
#         #     "stats_file"  : stats_file, 
#         #     "iscell"      : iscell, 
#         #     "stats"       : stats, 
#         #     "ops"         : ops, 
#         #     "db"          : db, 
#         #     "output_ops"  : output_ops, 
#         #     "trigger_arr" : trigger_arr,
#         #     "header_info" : header_info
#         #     }
#             # df = pd.DataFrame(data = data_dict)
#             # load_path.stem = data
#             # data_dump = data
#             # return load_path.stem, data
#             # print(data['f_cells'])