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
    
    if 'interpolation_coefficient' in kwargs:    
        interpolation_coefficient = kwargs['interpolation_coefficient']
    else:
        # interpolation_coefficient = 10000
        interpolation_coefficient = f.shape[1] * 10 #Upskale by x10
    
    trig_frames = trigger.nonzero()[0]
    first_trg_indx = trig_frames[0]
    repeats = len(trig_frames)/mode
    
    cropped_f, cropped_trig = utilities.data.crop(f, trigger, 0, 0)
    
    # Sometimes the f arrays are misalinged by a single frame.
    # The following algorithm handles this scenario by upsampling the data
    # to a specific temporal resolution (e.g., all arrays are 1000 frames).
    trig_frames = trigger.nonzero()[0]
    def interpolate_each_trace(f, mode, repeats, interpolation_granularity):
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
    
    sliced_traces  = interpolate_each_trace(f, 30, 3, interpolation_granularity = interpolation_coefficient)
    sliced_triggers = slice_triggers(trigger, interpolation_granularity = interpolation_coefficient)
    sliced_triggers = np.where(sliced_triggers>0, 1, 0)
    averaged_traces = np.average(sliced_traces, axis = 0)
    
    return averaged_traces, sliced_traces, sliced_triggers


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