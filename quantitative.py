# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 11:39:06 2022

@author: SimenLab
"""
import pathlib
import numpy as np 
import pandas as pd

# data_path = r"./Data/data_save/1_8dfp_ntc3_512_1.npz"

def load_data(path):
    load_path = pathlib.Path(path)
    data = np.load(load_path.with_suffix('.npz'), allow_pickle = True)
    return data

def data_struct(npz):
    npz.files
    
# exp_test = load_data(data_path)

def average_


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