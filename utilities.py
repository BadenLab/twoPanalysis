# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:00:15 2022

@author: SimenLab
"""

import pathlib
import numpy as np
import tifffile
import warnings
import shutil
import pickle
import os
    
class data:

    def crop(f, trigger, start_buffer, end_buffer):
        """
        Crop the 
    
        Parameters
        ----------
        f : TYPE
            DESCRIPTION.
        trigger : TYPE
            DESCRIPTION.
        start_buffer : TYPE
            DESCRIPTION.
        end_buffer : TYPE
            DESCRIPTION.
        tau : int
            The time interval, aka frames per seconds.
            
        Returns
        -------
        cropped_f : 2d-array
            Cropped flouresence traces
        cropped_trigger : 1d-array
            Cropped trigger signal
    
        """
        f_len               = f.shape[0]
        # seconds_to_frames   = 1/tau
        f_cropped           = f[start_buffer:f_len-end_buffer]
        trigger_cropped     = trigger[start_buffer:f_len-end_buffer]
        return f_cropped, trigger_cropped
    
    def interpolate(input_array, output_trace_resolution):
        """
        Interpolate
        
        Parameters
        ----------
        input_array : TYPE
            DESCRIPTION.
        output_trace_resolution : TYPE
            DESCRIPTION.

        Returns
        -------
        interpolated_trace : nd-array
            DESCRIPTION.

        """
        if input_array.ndim > 1 == True:
            if input_array.ndim == 2:
                interp_list = np.empty((len(input_array), output_trace_resolution))
                for n, trace in enumerate(input_array):
                    x = np.arange(0, len(trace))
                    y = trace
                    x_new = np.linspace(0, len(trace), output_trace_resolution)
                    interpolated_trace = np.interp(x_new, x, y)
                    
                    interp_list[n] = interpolated_trace
            if input_array.ndim == 3:
                interp_list = np.empty((input_array.shape[0], input_array.shape[1], output_trace_resolution))
                for n, array in enumerate(input_array):
                    for m, trace in enumerate(array):
                        x = np.arange(0, len(trace))
                        y = trace
                        x_new = np.linspace(0, len(trace), output_trace_resolution)
                        interpolated_trace = np.interp(x_new, x, y)
                        interp_list[n][m] = interpolated_trace
        
            # else:
            #     interp_list = np.empty((input_array.ndim, input_array.shape[1], output_trace_resolution))
            #     # print(input_array.shape)
            #     for n, array in enumerate(input_array):
            #         for m, trace in enumerate(array):
            #             x = np.arange(0, len(trace))
            #             y = trace
                        
            #             x_new = np.linspace(0, len(trace), output_trace_resolution)
            #             interpolated_trace = np.interp(x_new, x, y)
            #             # print(n, m)
            #             interp_list[n][m] = interpolated_trace
            #         # np.append(interpolated_trace, interp_list)
            
            
            return interp_list
        else:
            x = np.arange(0, len(input_array))
            y = input_array
            
            x_new = np.linspace(0, len(input_array), output_trace_resolution)
            interpolated_trace = np.interp(x_new, x, y)
        
            return interpolated_trace
    # Make trigger_trace
    def trigger_trace_by_frame(trigger_arr):
        raise DeprecationWarning("Calling trigger trace from Import_Igor is depricated. Instead, please call attribute trigger_trace_frame from Import_Igor.get_stack obj instead. This may change in the future.")
        #Binarise the trigger channel using frame-wise percision (fast but impercise)
        ## Make an array of appropriate dimension
        trigger_trace_arr = np.zeros((1, trigger_arr.shape[0]))[0]
        ## Loop through trigger image array
        for frame in range(trigger_arr.shape[0]):
            if np.any(trigger_arr[frame] > 1):
                trigger_trace_arr[frame] = 1
            #If trigger is in two consecutive frames, just use the first one so counting is correct
            if trigger_trace_arr[frame] == 1 and trigger_trace_arr[frame-1] == 1: 
                trigger_trace_arr[frame] = 0
        return trigger_trace_arr
    def trigger_trace_by_line(trigger_arr):
        raise DeprecationWarning("Calling trigger trace from Import_Igor is depricated. Instead, please call attribute trigger_trace_line from Import_Igor.get_stack obj instead. This may change in the future.")
        #Binarise the trigger channel using line-wise percision (slow but guaranteed percision)
        ## Make empty array that has dims frame_number x frame_size (for serialising each frame)
        trigger_trace_arr = np.empty((len(trigger_arr), trigger_arr[0].size))
        ## Loop through the input trigger array and serialise each frame
        for n, frame in enumerate(trigger_arr):
            serial = frame.reshape(1, frame.size)
            ## Place that serialised data in its correct index
            trigger_trace_arr[n] = serial
        ## Our matrix is now an array of vectors containing serialised information from each frame
        ## Reshape this matrix into one long array (pixel x pixel-value)
        serial_trigger_trace = trigger_trace_arr.reshape(1, trigger_arr.size)
        ## Then we binarise the serialised trigger data
        binarised_trigger_trace = np.where(serial_trigger_trace > 10000, 1, 0)[0]
        ## Boolean search for whether index n > n-1 (basically rising flank detection) 
        trig_onset_serial = binarised_trigger_trace[:-1] > binarised_trigger_trace[1:]
        ## Get the frame indeces for trigger onset
        trig_onset_index = np.where(trig_onset_serial > 0)
        ## Then divide each number in trig_onset_index by the amount of lines
        trigg_arr_shape = trigger_arr.shape
        lines_in_scan = trigg_arr_shape[1] * trigg_arr_shape[2]
        frame_of_trig = np.around(trig_onset_index[0]/lines_in_scan, 0)
        ## Convert back to frames 
        frame_number = len(trigger_arr)
        trig_trace = np.zeros(frame_number)
        for i in frame_of_trig:
            trig_trace[int(i)] = 1
        return trig_trace
    
# test2 = np.load(r"C:\Users\SimenLab\OneDrive - University of Sussex\Desktop\test.npy")
# v = interpolate(test2, 300)