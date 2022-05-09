# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:13:31 2022

@author: SimenLab
"""
# Environment imports:
import pathlib
import numpy as np 
import pandas as pd
# import sys

#Local imports:
# import Suite2me as s2m
import utilities
import quantitative
import qualitative
import classes

# fs, trig_trace = utilities.file_handling().load_experiment()

outpt = "test_experiment_out"
inpt = "test_experiment"

# Make it so that this class returns a pandas dataframe via a dictionary or something similar 
    
    
# location = pathlib.Path(r"C:\Users\Simen\OneDrive\Universitet\PhD\test_experiments")
location = pathlib.Path(r"D:\data_output\test_BC_out_AAA")
obj = Experiment(location, averages = True, mode = 30)
# f_avg, f_trial, trig_trial = quantitative.average_signal(obj.fs[0][0], obj.trigs[0], 30)
pand = obj.panda
experiment, roi = 1, 2
qualitative.plot_averages(pand.loc["f_avgs"][experiment], pand.loc["f_trials"][experiment], pand.loc["trig_trials"][experiment], roi)