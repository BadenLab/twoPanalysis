# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:13:31 2022

@author: SimenLab
"""
#Local imports:
import Suite2me as s2m

# Environment imports:
import pathlib


# params = Experiment.Stim_parameters(
#     Mode=20,
#     Repeats=3,
#     On_dur=4,
#     Off_dur=3,
#     Stim_width_um=100,
#     Cycle_len_s=7.0787,
#     LineDuration=0.001956
# ) # This does very little as of now

# File-handling related
def get_imaging_path(path):
    img_path = pathlib.Path(path)
    return img_path

inpt = "test_BCs"
outpt = "test_BCs"

# Work
ops = s2m.extract_singleplane(r"D:\data_input\{}".format(inpt),
                            r"D:\data_output",
                            "{}".format(outpt), 
                            256)



"""For testing on other computers"""
#Home
# ops = s2m.extract_data(r"D:\data_input\{}".format(inpt),
#                             r"D:\data_output",
#                             "{}".format(outpt),s
#                             512)

#Laptop
# s2m.extract_singleplane(r"C:\Users\Simen\Desktop\{}".format(inpt),
#                             r"C:\Users\Simen\Desktop",
#                             "{}".format(outpt),
#                             256)
#                             # r"C:\Users\Simen\Desktop\BC testing.npy")