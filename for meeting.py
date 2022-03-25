# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:19:13 2022

@author: SimenLab
"""

import qualitative
import numpy as np
import matplotlib.pyplot as plt

new_exp_test = np.load(r"C:\Users\SimenLab\OneDrive\Universitet\PhD\Python files\Git repos\2Panalysis - Copy\Data\old\Old failed attempts\1_8dfp_ntc3_512_1.npz")

trigger = qualitative.trigger_trace(new_exp_test["trigger_arr"])
qualitative.plot_heatmap(new_exp_test["f_cells"], trigger, colors = ['m', 'b', 'g', 'r'])

def plot_trig(trig_trace):
    plt.figure(dpi = 300, figsize=(10, 2))
    plt.plot(trigger)
    plt.title("Trigger ch")
    plt.show()
    
    