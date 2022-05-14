# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 11:44:31 2022

@author: SimenLab
"""

import matplotlib.pyplot as plt
import pathlib
import numpy as np 
import pathlib
import sklearn as sk
# scaler = sk.preprocessing.MinMaxScaler()

#Load data
def load_data(path):
    load_path = pathlib.Path(path)    
    with np.load(load_path.with_suffix('.npz')) as data:
        # load_path.stem = data
        data = data
        return load_path.stem, data




def display(im3d, cmap="gray", step=1):  # Not written by me:) --> https://scikit-image.org/docs/dev/auto_examples/applications/plot_3d_image_processing.html#sphx-glr-auto-examples-applications-plot-3d-image-processing-py
    _, axes = plt.subplots(nrows=22, ncols=21, figsize=(16, 14))

    vmin = im3d.min()
    vmax = im3d.max()

    for ax, image in zip(axes.flatten(), im3d[::step]):
        ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
    

def plot_heatmap(f, trigger, normalize = 1, **kwargs):
    plt.figure(figsize=(20, 10), dpi = 200)
    
    if normalize == 1:
        def normalize(data):
                data_normalized = (data - np.mean(data)) / np.std(data)
                return data_normalized
        norm_f_array = normalize(f)
        # norm_f_array = np.copy(f) #create duplicate to avoid overwriting original imported data matrix
        # for i in range(f.shape[1]): 
        #     curr_operation = scaler.transform((norm_f_array[:, i]).reshape(-1, 1)) #"""workaround"""
        #     curr_operation = curr_operation.reshape(len(curr_operation))
        #     norm_f_array[:, i] = curr_operation
            
        plt.imshow(norm_f_array, aspect = 'auto', cmap = 'Greys')
    else:
        plt.imshow(f, aspect = 'auto', cmap = 'Greys')   
    if "colors" in kwargs:
        color_list = kwargs["colors"]# ['m', 'b', 'g', 'r']
        # Make a simple counter 
        counter = 0
        # To plot lines for stimulus event
        for n, i in enumerate(trigger):
            # Where trigger channel has signal
            if trigger[n] == 1:
                # 
                if counter == 3:
                    plt.axvspan(n, n+1, facecolor=color_list[counter], alpha=0.5)
                    counter = 0
                else:
                    plt.axvspan(n, n+1, facecolor=color_list[counter], alpha=0.5)
                    counter += 1
    if "events" in kwargs:
        # To plot lines for stimulus event
        for n, i in enumerate(trigger):
            if trigger[n] == 1:
                    plt.axvspan(n, n+1)
    plt.show()
    
    return norm_f_array

    """This doesn't work either
    Try this: https://stackoverflow.com/questions/5628055/execute-statement-every-n-iterations-in-python
    """
    # for n, i in enumerate(trigger[::2]):
    #     if trigger[n] == 1:
    #         plt.axvspan(n, n+1, facecolor='g', alpha=0.5)
    

def plot_traces(output_ops, f_cells, f_neuropils, spks):
    plt.figure(figsize=[20,20])
    plt.suptitle("Fluorescence and Deconvolved Traces for Different ROIs", y=0.92);
    rois = np.arange(len(f_cells))[::20] # Sets how many of the ROIs to sample from for plotting
    for i, roi in enumerate(rois):
        plt.subplot(len(rois), 1, i+1, )
        f = f_cells[roi]
        f_neu = f_neuropils[roi]
        sp = spks[roi]
        # Adjust spks range to match range of fluroescence traces
        fmax = np.maximum(f.max(), f_neu.max())
        fmin = np.minimum(f.min(), f_neu.min())
        frange = fmax - fmin 
        sp /= sp.max()
        sp *= frange
        plt.plot(f, label="Cell Fluorescence")
        plt.plot(f_neu, label="Neuropil Fluorescence")
        plt.plot(sp + fmin, label="Deconvolved")
        plt.xticks(np.arange(0, f_cells.shape[1], f_cells.shape[1]/10))
        plt.ylabel(f"ROI {roi}", rotation=0)
        plt.xlabel("frame")
        if i == 0:
            plt.legend(bbox_to_anchor=(0.93, 2))



def plot_averages(f_avg, f_trial, trig_trial, trig_avg, roi): #, trigger, mode, ):
    """
    
    """
    """
    TODO
    Fix the plotting of trigger, as it now is interpolated and plots the entire
    "series" (which is not real, because trigger should just last a single frame)
    thereby making the plot confusing to read.
    
    Could average the onset of every trigger 
    """
    fig, ax1 = plt.subplots(figsize= (12, 8), dpi = 100)
    ax2 = ax1.twinx()
    
    # plt.figure()
    """
    NB
    The below line currently only plots the average of all triggers per trial, 
    binarised such that n<1 = 0
    """
    # trig_avg = np.average(trigs_for_trial, axis = 0)
    # trig_binary = np.where(trig_avg<1, 0, 1)
    beauty_trig = np.zeros((len(trig_avg)))
    for n, i in enumerate(trig_avg):
        if trig_avg[n-1] == 0 and trig_avg[n] == 1:
            beauty_trig[n] = 1
        if trig_avg[n-2] == 1 and trig_avg[n-1] == 1 and trig_avg[n] == 0:
            trig_avg[n] = 0
        ## Old
        # if trig_avg[n] == 1:
        #     beauty_trig[n] = 1
        # if trig_avg[n] == 1 and trig_avg[n-1] == 1: 
        #     trig_avg[n] = 0
        # else:
        #     trig_avg[n] = 0
    # ax2.plot(trig_avg)
    ## Find, on average, when the trigger occurs and plot at its onset
    # avg_trig_distance = round(np.average(np.gradient(trig_frames)))
    
    ax2.plot(beauty_trig)
    # ax2.plot(trig_binary)
    for i in range(f_trial.shape[0]):
        ax1.plot(f_trial[i][roi], color = 'lightgrey')
    ax1.plot(f_avg[roi], color = 'r')
    ax2.set_ylim(0, 1)
    
    plt.show()
     
    
def plot_trigger(trigger):
    fig, ax1 = plt.subplots(figsize= (12, 8), dpi = 1000)
    plt.plot(trigger)

#Vizualisation
# s2m.detection_viz(stats_file, iscell, stats, output_ops)
# s2m.plot_traces(output_ops, f_cells, f_neuropils, spks)

# new_exp_test = np.load(r"C:\Users\SimenLab\OneDrive\Universitet\PhD\Python files\Git repos\2Panalysis\Data\old\1_8dfp_ntc3_512_1.npz")
# 
# trigger = trigger_trace(new_exp_test["trigger_arr"])
# plot_heatmap(new_exp_test["f_cells"], trigger, colors = ['m', 'b', 'g', 'r'])
# plot_heatmap(new_exp_test["f_cells"], trigger, events = True)