# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 12:27:40 2021

@author: skrem
"""

import pandas as pd

def Avg_data_getter(file):
    """A function which gets and prepares data, as well as returning some
    additional params like an number of ROIs and their corresponding
    labels.
    
    Parameters
    ---------
    File: 
        The directory of a CSV .txt file containing the data, with each ROI
        represented as individual columns in the file.
    
    Returns
    -------
    Stimulus DataFrame, stimulus array (alt), number of ROIs, and their labels
    """

    avgs_data = pd.read_csv((file), sep="\t", header=None, engine = "python")
    
    ##Label data
    ROI_num  = avgs_data.shape[1]

    "Can put optional manipulations to data here" #Consider making data loading its own function? 
    
    averages_dataframe  = avgs_data #pd.read_csv((file), sep="\t", header=None) #, names = labels)
    avgerages_array  = pd.DataFrame.to_numpy(averages_dataframe)
    
    return averages_dataframe, avgerages_array, ROI_num