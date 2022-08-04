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
    via_interpolation : bool
        Determines whether to use interpolation to deal with mismatched frame-intervals...
    interpolation_granularity : int 
        Default: 100 x frame_number. The amount of points to generate after interpolation, independent of 
        what the original input is. Can be specified to any value (but should
        be used carefully...)

    Returns
    -------
    averaged_traces, sliced_traces, sliced_triggers

    """
    # Set up/correct some parameters
    if trigger.ndim == 2:
        trigger = trigger[0]
    if f.dtype != 'float64':
        f = f.astype(float)
    if trigger.dtype != 'float64':
        trigger = trigger.astype(float)

    ## Boolean search for whether index n > n-1 (rising flank)
    trig_onset_serial = trigger[:-1] > trigger[1:]
    ## Get locations of onset
    trig_onset_locs = np.where(trig_onset_serial == 1)[0]
    num_of_trigs = len(trig_onset_locs)
    repeats = int(num_of_trigs/mode)
    num_of_frames = f.shape[1]
    
    ## Compute average distance between triggers
    avg_trig_distance = np.average(np.gradient(trig_onset_locs))
    ## Compute how long a trial/repeat is on average
    avg_rep_distance = avg_trig_distance * mode
    
    print(f"{num_of_trigs} triggers and {repeats} repeats. F-array is {num_of_frames} long")

    ## Upscale F to be equal to trigger trace with line-precision
    f = utilities.data.interpolate(f, len(trigger))
    ## Pre-allocate memory for coding niceness
    repeats_list = np.empty([repeats, f.shape[0], round(avg_rep_distance)]) # shape is: repeats x cell_number x average inter-trigger distance
    trigger_list = np.empty([repeats, round(avg_rep_distance)])
    print("Averaging: ")
    ## For each repeat/trial of the experiment...
    for rep in range(repeats):
        ## get data from this index...
        from_index = trig_onset_locs[(rep)*mode]
        ## to this index
        to_index = from_index + round(avg_rep_distance)
        # to_index = trig_frames[(rep+1)*mode]
        print(" - From", from_index, "to", to_index, "for rep", rep)
        ## Index that data in both F and trigger trace
        activity_segment = f[:, from_index:to_index]
        trigger_segment = trigger[from_index:to_index]
        ## Add that to our list of repeats and triggers
        repeats_list[rep] = activity_segment
        trigger_list[rep] = trigger_segment
    ## Make averages for F and pass repeats_list as trial_traces (slightly redundant)
    averaged_traces   = np.average(repeats_list, axis = 0)
    trial_traces      = repeats_list
    ## Make averages for trig
    ## Do some voodoo to fix average trigger channel --> Extract trigger locations and re-construct trigger trace
    ### Make a list to keep track of where triggers fall in each segment (array is better)
    trigger_locs = []
    ### Loop through the list of triggers 
    for n, i in enumerate(trigger_list):
        trg_lc = np.where(i > 0)[0]
        ## Exception handling for if trigger alignment is off by a few lines
        if len(trg_lc) > mode:
            ## Simply get rid of the last trigger (e.g. grab data between 0 and 'mode' indeces) 
            trg_lc = trg_lc[:mode]
        ## Append this to the list to keep track where triggers fell in that segment
        trigger_locs.append(trg_lc)
    ## Get the average trigger location
    avg_trig_loc = np.around(np.average(trigger_locs, axis = 0), 0).astype(int)
    averaged_triggers = np.empty(round(avg_rep_distance))
    for i in (avg_trig_loc):
        averaged_triggers[i] = 1
    ## Pass trigger_list as trial_triggers
    trial_triggers    = trigger_list
    ### Binarise triggers as needed
    # trial_triggers = np.where(trial_triggers>0, 1, 0) # Binarise 
    # averaged_triggers = np.where(averaged_triggers>0, 1, 0) # Binarise 
    return averaged_traces, trial_traces, trial_triggers, averaged_triggers


    # if 'via_interpolation' in kwargs and kwargs['via_interpolation'] is True:

    #     # first_trg_indx = trig_frames[0]

    #     ## Add an "estimated" trigger to the end of trig_frames (such that we also get the last response)
    #     value_to_add = trig_frames[-1] + avg_trig_distance
    #     trig_frames = np.append(trig_frames, value_to_add)
    #     if 'interpolation_coefficient' in kwargs:    
    #         interpolation_coefficient = kwargs['interpolation_coefficient']
    #     else: #Upscale by this value
    #         interpolation_coefficient = num_of_frames * 100 
    #     # Sometimes the f arrays are misalinged by a single frame, due to triggers
    #     # occasionally being temporally misaligned. The following algorithm handles
    #     # this scenario by upsampling the data to a specific temporal resolution
    #     # (e.g., all arrays are 1000 frames) then downscaling again.
    #     def interpolate_each_trace(f, mode, repeats, interpolation_granularity):
    #         # Create empty array with the correct shape
    #         loops_list = np.empty([repeats, f.shape[0], interpolation_granularity])
    #         print("Averaging: ")
    #         for rep in range(repeats):
    #             from_index = trig_frames[(rep)*mode]
    #             to_index = trig_frames[(rep+1)*mode]
    #             print("From", from_index, "to", to_index, "for rep", rep)
    #             activity_segment = f[:, from_index:to_index]
    #             interpolated_activitiy_segment = utilities.data.interpolate(activity_segment, output_trace_resolution = interpolation_granularity)
    #             loops_list[rep] = interpolated_activitiy_segment
    #         return loops_list
    #     def slice_triggers(c, mode, repeats, interpolation_granularity):
    #         trig_list = np.empty([round(repeats), interpolation_granularity])
    #         for rep in range((repeats)):
    #             from_index = trig_frames[(rep)*mode]
    #             to_index = trig_frames[(rep+1)*mode]
    #             trigger_segment = trigger[from_index:to_index]
    #             interpolated_trig_segment = utilities.data.interpolate(trigger_segment, output_trace_resolution = interpolation_granularity)
    #             trig_list[rep] = interpolated_trig_segment
    #         return trig_list
        
    #     # Interpolate and slice as needed
    #     trial_traces  = interpolate_each_trace(f, mode, repeats, interpolation_granularity = interpolation_coefficient)
    #     averaged_traces = np.average(trial_traces, axis = 0)
    #     trial_triggers = slice_triggers(trigger, mode, repeats, interpolation_granularity = interpolation_coefficient)
    #     averaged_triggers = np.average(trial_triggers, axis = 0)
    #     # Interpolate back to oringinal temporal resolution 
    #     trial_traces = utilities.data.interpolate(trial_traces, int(num_of_frames/repeats)) 
    #     averaged_traces = utilities.data.interpolate(averaged_traces, int(num_of_frames/repeats)) 
    #     trial_triggers = utilities.data.interpolate(trial_triggers, int(num_of_frames/repeats))
    #     averaged_triggers = utilities.data.interpolate(averaged_triggers, int(num_of_frames/repeats)) 
    #     # Binarise trigger
    #     trial_triggers = np.where(trial_triggers>0, 1, 0) # Binarise 
    #     averaged_triggers = np.where(averaged_triggers>0, 1, 0) # Binarise 
