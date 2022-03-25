# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 11:48:13 2021

@author: skrem
"""
import numpy as np
import settings_init
from Avg_data_getter import Avg_data_getter

if settings_init.storage_location is not None:
    "Settings auto-generated from input parameters_______________________________"
    avg_df = Avg_data_getter(settings_init.file_location)[0]
    avg_array = Avg_data_getter(settings_init.file_location)[1]
    ROI_number = Avg_data_getter(settings_init.file_location)[2]
    
    "Calculate rest of paramaters and set some handy variables___________________"
    ##Calculated parameters
    #   Visual angle calculations
    um_in_mm = 0.02 #50 um is 1mm, 0.02um is 1mm
    stim_width_mm = um_in_mm * settings_init.Stim_width_um
    fish_screen_distance = 16  #mm
    vis_ang_circumfrance = fish_screen_distance * np.pi * 2
    stim_width_visang = round(((stim_width_mm/vis_ang_circumfrance) * 360), 3)
    vis_ang_list = np.linspace(0, stim_width_visang*settings_init.Mode/2, int((settings_init.Mode/2)+1)) #0 to x
    vis_ang_list_alt = np.linspace(-stim_width_visang*settings_init.Mode/4, stim_width_visang*settings_init.Mode/4, int((settings_init.Mode/2)+1)) #-x/2 to 0 to x/2
    
    #  Time
    total_time  = len(avg_df)*settings_init.LineDuration
    indeces_per_s = avg_df.shape[0] / total_time #Amount of indeces for 1s 
    total_time_in_index = indeces_per_s * total_time
    
    # ON_phase_ratio = settings_init.On_len_s/settings_init.Trg_t_slope   ## Depricated (potentially)
    # OFF_phase_ratio = settings_init.Off_len_s/settings_init.Trg_t_slope ## Depricated
    
    index_list = np.arange(0, len(avg_df))
    seconds_list = np.arange(0, total_time, settings_init.LineDuration)
    
    ##Analysis params
    response_avg_dur = settings_init.On_len_s * .33 #last amount of seconds for which to infer peak response or baseline
    baseline_avg_dur  = settings_init.Off_len_s * .33