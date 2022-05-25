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
from matplotlib.pyplot import cm
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
    fig = plt.figure(figsize=(2,4), dpi = 200)
    
    if normalize == 1:
        def normalize(data):
            return (data - np.mean(data)) / np.std(data)
        norm_f_array = normalize(f)
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
        if "manual_trigger" in kwargs:
            trigger = kwargs["manual_trigger"]
        # To plot lines for stimulus event
        for n, i in enumerate(trigger):
            if trigger[n] == 1:
                    plt.axvspan(n, n+0.1)
    plt.show()
    return fig

    # """This doesn't work either
    # Try this: https://stackoverflow.com/questions/5628055/execute-statement-every-n-iterations-in-python
    # """
    # for n, i in enumerate(trigger[::2]):
    #     if trigger[n] == 1:
    #         plt.axvspan(n, n+1, facecolor='g', alpha=0.5)
    

def plot_traces_outdated(f_cells, spks): #f_neuropils, spks):
    fig = plt.figure(figsize=[8,8])
    plt.suptitle("Fluorescence and Deconvolved Traces for Different ROIs", y=0.92);
    rois = np.arange(len(f_cells)) # Sets how many of the ROIs to sample from for plotting
    for i, roi in enumerate(rois):
        fig, axs = plt.subplot(len(rois), 1, i+1, )
        f = f_cells[roi]
        # f_neu = f_neuropils[roi]
        # sp = spks[roi]
        # Adjust spks range to match range of fluroescence traces
        # fmax = np.maximum(f.max(), f_neu.max())
        # fmin = np.minimum(f.min(), f_neu.min())
        # fmax = f.max()
        # fmin = f.min()
        # frange = fmax - fmin 
        # sp /= sp.max()
        # sp *= frange
        axs = plt.plot(f, label="Cell Fluorescence")
        # plt.plot(f_neu, label="Neuropil Fluorescence")
        # plt.plot(sp + fmin, label="Deconvolved")
        axs.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.xticks(np.arange(0, f_cells.shape[1], f_cells.shape[1]/10))
        plt.ylabel(f"ROI {roi}", rotation=0)
        plt.xlabel("frame")
        if i == 0:
            plt.legend(bbox_to_anchor=(0.93, 2))
            
def plot_traces(fs, from_num, to_num):
    f_cells = fs[from_num:to_num]
    num_to_plot = len(f_cells)
    num_index = np.arange(from_num, to_num)
    color = iter(cm.rainbow(np.linspace(0, 1, num_to_plot)))
    fig, axs = plt.subplots(len(f_cells), figsize=(5,5), sharex = True)
    # fig.suptitle('Vertically stacked subplots')
    for n, i in enumerate(f_cells):
        c = next(color)
        axs[n].plot(i, c = c)
        axs[n].spines['top'].set_visible(False)
        axs[n].spines['bottom'].set_visible(False)
        axs[n].spines['left'].set_visible(False)
        axs[n].spines['right'].set_visible(False)
        # axs[n].axes.get_yaxis().set_visible(False)
        axs[n].set_yticklabels([])
        axs[n].set_yticklabels([])
        axs[n].set_yticks([])
        axs[n].axes.get_xaxis().set_visible(False)
        axs[n].set_ylabel(f"ROI {num_index[n]}", rotation = 'horizontal',
                          fontsize = 'x-small', va = 'center_baseline')
    return plt.show()
# plot_traces(pand.loc["fs"][0], 10, 20)
# plot_traces(pand.loc["fs"][0], pand.loc["spks"][0])

def plot_averages(f_avg, f_trial, trig_trial, trig_avg, roi): #, trigger, mode, ):
    fig, ax1 = plt.subplots(figsize= (12, 8), dpi = 100)
    ax2 = ax1.twinx()
    beauty_trig = np.zeros((len(trig_avg)))
    for n, i in enumerate(trig_avg):
        if trig_avg[n-1] == 0 and trig_avg[n] == 1:
            beauty_trig[n] = 1
        if trig_avg[n-2] == 1 and trig_avg[n-1] == 1 and trig_avg[n] == 0:
            trig_avg[n] = 0
    ax2.plot(trig_avg)

    ## Find, on average, when the trigger occurs and plot at its onset
    # avg_trig_distance = round(np.average(np.gradient(trig_frames)))
    for i in range(f_trial.shape[0]):
        ax1.plot(f_trial[i][roi], color = 'lightgrey')
    ax1.plot(f_avg[roi], color = 'r')
    ax2.set_ylim(0, 1)
    ax2.set_axis_off()
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