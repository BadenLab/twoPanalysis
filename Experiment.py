# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 13:59:20 2021

@author: skrem
"""

"""Prime per-experiment params"""

import pathlib
import numpy as np
from datetime import datetime
import os

#==== Pathlib =======
curr_path = pathlib.Path.cwd()
this_path = pathlib.Path(__file__).resolve().parent
os.chdir(this_path)
# print(os.getcwd())
#==== Pathlib =======

import Import_Igor as Igor

class Stim_parameters(object):
    def __init__(self, Mode, Repeats, On_dur, Off_dur, Stim_width_um,
                 Cycle_len_s, LineDuration):
    #neccesary settings
        self.Mode = Mode
        self.Repeats = Repeats
        self.On_dur = On_dur
        self.Off_dur = Off_dur
        self.Stim_width_um = Stim_width_um
        # self.Trg_t_slope = Trg_t_slope ## Depricated
        self.Cycle_len_s = Cycle_len_s
        self.LineDuration = LineDuration

class storage_info:
    def __init__(self, file_location):
        self.file_location_str = file_location
        self.experiment_path = pathlib.Path(file_location)
        self.file_parents = self.experiment_path.parents # get using [x] to iterate through path

class time_info:
    def __init__(self, parameter_object, data_object):
        self.frame_time_s = parameter_object.LineDuration * data_object.frameH
        self.frames_per_s = 1 / self.frame_time_s
        frame_num = len(data_object.channel1)
        self.total_time_s = self.frame_time_s * frame_num
        created = os.stat(data_object.file_path).st_ctime
        self.datetime = datetime.fromtimestamp(created)
        # self.oclock =
        # self.total_time_in_index = frame_num

        # self.total_time  = data_object.frameN * parameter_object.LineDuration * data_object.frameH
        # self.indeces_per_s = data_object.frameN / self.total_time #Amount of indeces for 1s
        # self.total_time_in_index = self.indeces_per_s * self.total_time

        "should fix this, something is screwy with the datetime scripts"

class space_info:
    def __init__(self, parameter_object):
        #   Visual angle calculations
        um_in_mm = 0.02 #50 um is 1mm, 0.02um is 1mm
        self.stim_width_mm = um_in_mm * parameter_object.Stim_width_um
        fish_screen_distance = 16  #mm
        vis_ang_circumfrance = fish_screen_distance * np.pi * 2
        self.stim_width_visang = round(((self.stim_width_mm/vis_ang_circumfrance) * 360), 3)
        self.vis_ang_list = np.linspace(0, self.stim_width_visang*parameter_object.Mode/2, int((parameter_object.Mode/2)+1)) #0 to x
        self.vis_ang_list_alt = np.linspace(-self.stim_width_visang*parameter_object.Mode/4, self.stim_width_visang*parameter_object.Mode/4, int((parameter_object.Mode/2)+1)) #-x/2 to 0 to x/2

class initiate:
    """
    Pass Stim_parameters class and Storage_info class, with corresponding params
    """
    def __init__(self, stim_params, file_path, **kwargs):
        if 'condition' in kwargs:
            if type(kwargs['condition']) is str:
                self.condition = [kwargs['condition']]
            if type(kwargs['condition']) is list:
                self.condition = kwargs['condition']
            else:
                raise TypeError("Keyword argument 'condition' must be of type str (single) or list (single/multiple).")

        # Inherit classes:
        ## Parameters
        self.stim_params = stim_params
        ## Storage
        self.storage = storage_info(file_path)
        ## Data
        experiment_path = pathlib.Path(file_path)
        experiment_filename = experiment_path.name
        experiment_parent_folder =  experiment_path.parent
        self.data = Igor.get_stack(experiment_parent_folder, experiment_filename) # Inherited class-object with data information
        ## Time
        self.time = time_info(self.stim_params, self.data)
        ## Space
        self.space = space_info(self.stim_params)

        # if isinstance(self.storage_info, Storage_info) is not True:
        #     raise TypeError("Argument 'storage_info' must be instance of class Storage_info.")
        # if isinstance(self.stim_params, Stim_parameters) is not True:
        #     raise TypeError("Argument 'stim_params' must be instance of class Stim_parameters.")


"""
_______________Testing_________________________________________________________
"""
# params = Stim_parameters(
#                 Mode = 20,
#                 Repeats = 3,
#                 On_dur = 4,
#                 Off_dur = 3,
#                 Stim_width_um = 100,
#                 Cycle_len_s = 7.0787,
#                 LineDuration = 0.001956
#                 )

# experiment_file_path = r'C:\Users\Simen\Test data folder\L5-20my l10x10 100um.smh'
# experiment_file_path = r"Z:\Data\Bruoygard\20211207\sb_naturalistic_brain4.smp"
# exp1 = Experiment(params, experiment_file_path) #conditions = conds)

# convert_to_tif(exp1.data.channel1, r"C:\Users\SimenLab\OneDrive - University of Sussex\Desktop", "test")
# tiff.convert_to_tif(exp1.data.channel1, r"C:\Users\SimenLab\OneDrive - University of Sussex\Desktop", "test2.tiff")

"""
TODO
- [x] Fix border in img files
    - Currently crops at final processing stage (See get_ch_arrays in Import_Igor)
    - [ ] Should move this to somewhere more logical
- [ ] Check that timing info is accurate (compare to CSV method...)
- [ ] Make Stim_parameters class to accept any amount of kwargs. Pass these to a dictionary and make all .self for the object
- [ ] Add error messages if Experiment.[objects] are incorrect classes (e.g. object 'data' is not from class 'Igor.get_stack')
- [ ] Make a script that creates experiment objects by crawling through folder with .smh and .smp files, runs them through Suite2p,
and saves each experiment
"""

# ------------------- Leftovers ---------------------------------------------


# img1 = Import_Igor.get_stack(directory, "L5-20my l10x10 100um")

# store = Storage_info("C://Users//SimenLab//Desktop//test export.txt")
# cond = ["Retina"]

# exp1 = Experiment(params, store, condition = cond)

# directory = r'C:\Users\Simen\Test data folder'
# store = Storage_info(r'C:\Users\Simen\Test data folder')


# Stim_settings = Stim_parameters(20, 3, 4, 3, 100, 7.0787, 0.001956)

##Legacy stim params
# Mode = 20
# repeats = 0 #optional
# On_len_s = 4
    # Off_len_s = 3
# Stim_width_um = 100
# Trg_t_slope = 7.0787
# Cycle_ len_s = Trg_t_slope
# LineDuration = 0.001956

## Legacy storage info
# # storage_location = "D:\Dissertation files\Further analysis"
# storage_location = None
# stim_type_folder = "Lines"
# resolution_folder = "{} x {} 100um".format(int(Settings.Mode/2), int(Settings.Mode/2))
# condition_folder = "Peripheral w CNQX"
# filename = "14.7 L2 10x10 100" # ROI 0
# file_location = r"{}\{}\{}\{}\{}.txt".format(storage_location, stim_type_folder, resolution_folder, condition_folder, filename)

## Legacy conditions list
# conds_list = ['AZ', 'AZ (CNQX)', 'Peripheral', 'Peripheral (CNQX)'] #semi-optional
