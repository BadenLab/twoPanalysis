# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:00:15 2022

@author: SimenLab
"""

import pathlib
import numpy as np

def load_experiment(f_path, trigger_path):
    f = np.load(f_path, allow_pickle = True)
    trigger = np.load(trigger_path, allow_pickle = True)
    return f, trigger

def get_content(folder_path):
    """
    

    Parameters
    ----------
    folder_path : TYPE
        DESCRIPTION.

    Returns
    -------
    folder_contents : TYPE
        DESCRIPTION.

    """
    folder = pathlib.Path(folder_path)
    folder_contents = list()
    for child in folder.iterdir():
        print(child)
        folder_contents.append(child)
    # content = folder.iterdir()
    return folder_contents

def get_ops(path):
    ops =  np.load(path, allow_pickle=True)
    ops = ops.item()
    
def read_ops(file_path):
    """
    

    Parameters
    ----------
    file_path : TYPE
        DESCRIPTION.

    Returns
    -------
    ops : TYPE
        DESCRIPTION.

    """
    ops = np.load(file_path, allow_pickles=True)
    ops = ops.item()
    return ops


def interpolate(input_array, output_trace_resolution):
    if input_array.ndim > 1 == True:
        if input_array.ndim == 2:
            interp_list = np.empty((len(input_array), output_trace_resolution))
            for n, trace in enumerate(input_array):
                x = np.arange(0, len(trace))
                y = trace
                
                x_new = np.linspace(0, len(trace), output_trace_resolution)
                interpolated_trace = np.interp(x_new, x, y)
                
                interp_list[n] = interpolated_trace
        else:
            interp_list = np.empty((input_array.ndim, input_array.shape[1], output_trace_resolution))
            for n, array in enumerate(input_array):
                for m, trace in enumerate(array):
                    x = np.arange(0, len(trace))
                    y = trace
                    
                    x_new = np.linspace(0, len(trace), output_trace_resolution)
                    interpolated_trace = np.interp(x_new, x, y)
                    
                    interp_list[n][m] = interpolated_trace
                # np.append(interpolated_trace, interp_list)
        return interp_list
    else:
        x = np.arange(0, len(input_array))
        y = input_array
        
        x_new = np.linspace(0, len(input_array), output_trace_resolution)
        interpolated_trace = np.interp(x_new, x, y)
    
        return interpolated_trace
# test2 = np.load(r"C:\Users\SimenLab\OneDrive - University of Sussex\Desktop\test.npy")
# v = interpolate(test2, 300)