# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 13:59:20 2021

@author: skrem
"""

"""Prime the global vars"""
#neccesary settings
Mode = 20
On_len_s = 4
Off_len_s = 3
Stim_width_um = 100
Trg_t_slope = 7.0787
Cycle_len_s = Trg_t_slope
LineDuration = 0.001956

#additional settings
conds_list = ['AZ', 'AZ (CNQX)', 'Peripheral', 'Peripheral (CNQX)'] #semi-optional
repeats = 0 #optional

#file location "hack"/settings
storage_location = "D:\Dissertation files\Further analysis"
stim_type_folder = "Lines" 
resolution_folder = "{} x {} 100um".format(int(Mode/2), int(Mode/2))
condition_folder = "Peripheral w CNQX"
filename = "14.7 L2 10x10 100" # ROI 0
file_location = r"{}\{}\{}\{}\{}.txt".format(storage_location, stim_type_folder, resolution_folder, condition_folder, filename)

#import path setting
import_path = r'C:\Users\skrem\OneDrive\Universitet\MSc\Experimental project\Data export'#optional
