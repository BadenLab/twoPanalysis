# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 20:37:15 2021

@author: skrem
"""

import pandas as pd
import numpy as np
# import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.preprocessing
from sklearn import metrics
import scipy.stats
import scipy.optimize
import seaborn as sns
import matplotlib.patheffects as path_effects
import os 
import copy
scaler = sk.preprocessing.MinMaxScaler()

degree_sign = u'\N{DEGREE SIGN}'

"Get global params and pass them to locals"
import settings_init
import settings_transformations
from Avg_data_getter import Avg_data_getter



if settings_init.storage_location is not None:
    file_location = settings_init.file_location
    Mode = settings_init.Mode
    On_len_s = settings_init.On_len_s
    Off_len_s = settings_init.Off_len_s
    Cycle_len_s = settings_init.Cycle_len_s
    repeats = settings_init.repeats
    Stim_width_um = settings_init.Stim_width_um
    conds_list = settings_init.conds_list
    
    response_avg_dur = settings_transformations.response_avg_dur
    baseline_avg_dur = settings_transformations.baseline_avg_dur
    indeces_per_s = settings_transformations.indeces_per_s
    total_time = settings_transformations.total_time
    vis_ang_list = settings_transformations.vis_ang_list
    seconds_list = settings_transformations.seconds_list
    
    avg_df = settings_transformations.avg_df
    avg_array = settings_transformations.avg_array
    ROI_number = settings_transformations.ROI_number


    "Functions____________________________________________________________________"
    
    
    def Get_event_data(roi = "All", event = "All", normalize = "0", plot = "0", data = file_location):
        """Returns a data for selected events specified (based on Mode), and computes 
        response and baseline average. 
        
        Hint: To select multiple ROIs for a single event or multiple events from a 
        single ROI, specify as variable eg.g ROI_13_14_15_event_8 = 
        Get_avg_response((13, 14, 15), (8)). Selecting both multiple ROIs and 
        multiple events is unstable and will yield unexpected results.
             
        Parameters
        ----------
        roi_select: Tuple or array 
             ROIs from which data is extracted. Default loops through all ROIs. 
             Script written to be naive to wheter input is tuple (one ROI) or
             array (many ROIs)
        event_select: Tuple or array 
            Events from which data is extracted. Default loops through all events.
            Naive to tuple (one event) or arrays (many events)
        normalize : 0 or 1
            Normalize data so range is from 0 to 1 (no/yes)
        plot: 0 or 1
            Plot sampled data 
        *data: If given (as string to directory), script loads new, external datafile
            
        Returns
        -------
        ROI_responses, ROI_baselines, Average_response, Average_baseline
        """
        
        # if data != file_location: 
        """
        TODO
        - This is not the neatest solution... IF I am to do this, then I should 
        seriously change the label to NOT BE THE SAME AS GLOBAL PARAMS. What I am
        doing currently is just a bit nasty...
        """
        alt_data = Avg_data_getter(data)
        avg_df = alt_data[0]     #"""A test"""
        avg_array = alt_data[1]
        ROI_number = alt_data[2]
        # label_list = alt_data[3]
        
        #new improvements
        
        if roi == "All":
            roi = np.arange(0, ROI_number)
        else: 
            roi = roi
        if isinstance(roi, int) == True:
            roi = np.array([roi])
            # print("roi was int(), converted to numpy array")
            #print("Warning: 'roi_select' takes tuple, but single int was given. Single int was converted to (1,) array.")
        if event == "All":
            event = np.arange(0, Mode)
        else:
            event = event
        if isinstance(event, int) == True:
            event = np.array([event])
            # print("event was int(), converted to numpy array")
            #print("Warning: 'event_select' takes tuple, but single int was given. Single int was converted to (1,) array.")
                            
            
        ROI_responses = np.empty((0,1))
        ROI_baselines = np.empty((0,1))
    
        if normalize == 1:
            norm_avg_array = np.copy(avg_array) #create duplicate to avoid overwriting original imported data matrix
            for i in roi: 
                """
                TODO
                - Fix the thing below... This is whats giving IndexError index 8 is out of bounds for axis 1 with size 8
                = what happens is that as loop starts, for some reason, it gets to a certain recording and index is 
                out of bounds for the ROIs in the recording...
                """
                curr_operation = scaler.fit_transform((norm_avg_array[:, i]).reshape(-1, 1)) #"""workaround"""
                curr_operation = curr_operation.reshape(len(curr_operation))
                norm_avg_array[:, i] = curr_operation
            normalized_data_set = pd.DataFrame(data = norm_avg_array, columns = np.arange(0, ROI_number))        
            data_set = normalized_data_set
        else: 
            data_set = pd.DataFrame.copy(avg_df)
        
        for i in roi:                                                                           #This script samples and extracts data at given intervals
            for j in event:
                #Get response values:
                start_index_res = (On_len_s - response_avg_dur + (Cycle_len_s * j)) * indeces_per_s    #set start position for current sampling
                end_index_res =   (On_len_s  + (Cycle_len_s * j)) * indeces_per_s                      #end position for current sampling
                                         
                curr_series_res = ((data_set[i].loc[start_index_res:end_index_res]))
                curr_series_res = curr_series_res.to_numpy()
                ROI_responses = np.append(curr_series_res, ROI_responses)
                #Get baseline values:
                start_index_bsl = (Cycle_len_s - baseline_avg_dur + (Cycle_len_s * j)) * indeces_per_s 
                end_index_bsl = (Cycle_len_s  + (Cycle_len_s * j)) * indeces_per_s  
                
                curr_series_bsl = ((data_set[i].loc[start_index_bsl:end_index_bsl]))
                curr_series_bsl = curr_series_bsl.to_numpy()
                ROI_baselines = np.append(curr_series_bsl, ROI_baselines)
        
        Average_response = np.average(ROI_responses)
        Average_baseline = np.average(ROI_baselines)
        
        if plot == 1:
            if len(roi) == 1:
                base_colors = mpl.cm.get_cmap('gist_rainbow')
                color_list = base_colors(np.linspace(0, 1, ROI_number))
                ROI_color = color_list[int(roi)]
            else:
                ROI_color = 'b'
            
            fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True, figsize = (10, 5))
            plt.subplots_adjust(wspace = 0)
            if isinstance(roi, int) == True:
                plt.suptitle("Sampled activity for ROI {}, event {}".format(int(roi), int(event)))
            else: 
                plt.suptitle("Sampled activity for ROIs {}, event {}".format((roi), (event)))
            
            # plt.figure(0)
            ax1.set_title("Response period")
            if normalize == 0: 
                ax1.set_ylabel("Z-score (raw)")
            if normalize == 1:
                ax1.set_ylabel("Z-score (normalised)")
            ax1.set_xlabel("Sample sequence")
            ax1.plot(ROI_responses, c = ROI_color)
            
            # plt.figure(1)
            ax2.set_title("Baseline period")
            # ax2.set_ylabel("Z-score")
            ax2.set_xlabel("Sample sequence")
            ax2.plot(ROI_baselines, c = ROI_color)
            #plt.vlines(np.linspace(0, len(ROI_resp_array.flatten('F')), Mode), np.amin(ROI_resp_array), np.amax(ROI_resp_array), colors = 'k')
        
        # print("Avg respone: {}, Avg baseline: {}".format(Average_response, Average_baseline))
        return ROI_responses, ROI_baselines, Average_response, Average_baseline
    
    def Get_interval_data(roi, interval_start_s, interval_end_s, normalize = "0", plot = "0"): 
        """Returns data from given ROI within specified time interval (s)
            
        Parameters
        -------------
        roi: int
            Which ROI to sample data from. Only one can be chosen at a time.
        interval_start_s: int
            Start of sampling interval (in seconds)
        interval_end_s: int
            End of sampling interval (in seconds)
        normalize : 0 or 1
            Normalize data so range is from 0 to 1 (no/yes)
        plot: 0 or 1
            Plot sampled data
        Returns
        -------
        interval_data, interval_data_with_s
        """
    
        if normalize == 1:
            norm_avg_array = np.copy(avg_array) #create duplicate to avoid overwriting original imported data matrix
            curr_operation = scaler.fit_transform((norm_avg_array[:,roi]).reshape(-1, 1)) #"""workaround"""
            curr_operation = curr_operation.reshape(len(curr_operation))
            norm_avg_array[:, roi] = curr_operation
            normalized_data_set = pd.DataFrame(data = norm_avg_array, columns = np.arange(0, ROI_number)) #np.arange(0, ROI_number)
            data_set = normalized_data_set
        else: 
            data_set = pd.DataFrame.copy(avg_df)
    
        interval_data = np.empty((0,1))
    
        start_index = interval_start_s * indeces_per_s    #set start position for current sampling
        end_index =   interval_end_s * indeces_per_s                      #end position for current sampling
    
        curr_series_res = ((data_set[roi].loc[start_index:end_index]))
        curr_series_res = curr_series_res.to_numpy()
        interval_data = np.append(curr_series_res, interval_data)
    
        if interval_end_s > total_time:
            time_in_s = np.linspace(interval_start_s, total_time, len(interval_data))
        else:
            time_in_s = np.linspace(interval_start_s, interval_end_s, len(interval_data))
        interval_data_with_s = np.column_stack((interval_data, time_in_s))
    
        if plot == 1:
            if isinstance(roi, int) is True:
                base_colors = mpl.cm.get_cmap('gist_rainbow')
                color_list = base_colors(np.linspace(0, 1, ROI_number))
                ROI_color = color_list[roi]
            else:
                ROI_color = 'b'
    
            plt.figure(0, dpi = 800)
            if normalize == 0: 
                plt.ylabel("Z-score (raw)")
            if normalize == 1:
                plt.ylabel("Z-score (normalised)")
            plt.title("Sampled interval data from ROI{}".format(roi))
            x_axis = time_in_s
            plt.plot(x_axis, interval_data, c=ROI_color)
            plt.xlabel("Time (s)")
            for m in range(Mode):
                plt.axvspan((m * Cycle_len_s), ((m * Cycle_len_s) + On_len_s),
                            color='r', alpha=0.25, lw=0)
                
            if interval_end_s > total_time:
                plt.xlim([interval_start_s, total_time])
            else: 
                plt.xlim([interval_start_s, interval_end_s])
            
        return interval_data, interval_data_with_s
    
    
    def Plot_activity(ROIs = "All", shade = 1, **kwargs):
        """Plot activity of all or specified ROIs"""
        if ROIs == "All":
            to_plot = np.arange(0, ROI_number)
        else:
            to_plot = np.array(ROIs)
    
        #Colormap
        base_colors = mpl.cm.get_cmap('gist_rainbow') #hsv(x) for x in range(ROI_number)] <-- legacy solution
        color_list = base_colors(np.linspace(0, 1, ROI_number))
        
        #Calculate time interval for x-axis
        time_in_s = np.linspace(0, total_time, len(avg_df))
        
        #Build each individual ROI plot
        # if ROIs == "All":
        fig, ax1 = plt.subplots(len(to_plot), 1, sharex = 'col', sharey = False, dpi = 1200, figsize=(10, 15))
        # else:
            # fig, ax1 = plt.subplots(len(to_plot), 1, sharex = 'col', sharey = False, dpi = 800, figsize=(10, 15))  
        for v, i in enumerate(to_plot):
            w = v+1
            ax1[v] = plt.subplot(len(to_plot), 1, w)
            ax1[v].plot(time_in_s, avg_df[i], color = color_list[i], linewidth=1.5)
            sns.despine(left = True, right = True, bottom = True)
            ax1[v].get_yaxis().set_visible(False)
            ax1[v].set_title("ROI{}".format(i), x=-0.01, y=.5, size = 10)
            
            
            if shade == 1:
                for m in range(Mode):
                            ax1[v].axvspan(
                            (m * Cycle_len_s), ((m * Cycle_len_s) + On_len_s),
                            color = '#ffe0f9', lw = 0)#, alpha = 0) 
            # plt.setp(ax1[i-1].get_xticklabels(), visible=False) #This is a work around. Hides axis 
                                                                #for every ax1 except last one, as                                                     #share-axis did not function properly.
        
        plt.subplots_adjust(hspace = 0)    
        
        #Frame for adding titles and such
        fig.add_subplot(111, frameon = False)
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Time (s)")
        plt.title("Average ROI activity ({} trials)".format(repeats))
    
        # ax2.spines["top"].set_visible(True)
        # ax2.spines["bottom"].set_visible(False)
        # ax2.spines["left"].set_visible(False)
        # ax2.spines["right"].set_visible(False)
        # ax2.axis("off")
        
        if 'saveas' in kwargs:
            # plt.figure(dpi = 2000)
            plt.savefig(r'C://Users//skrem//OneDrive//Universitet//MSc//Experimental project//Figures//Python generated//{}'.format(kwargs['saveas']), dpi = 2000,  bbox_inches='tight')
        
        
        
        plt.figure(2, dpi=800)
        ROI_overlap, bx = plt.subplots(1, 1, figsize=(15, 10))
        bx.set_title("All ROI activity")
        plt.locator_params(axis = 'x', tight = None, nbins = 30)
        for i in to_plot:
            bx.plot(seconds_list, avg_df[i], color = color_list[i], linewidth=0.75)
            bx.set_xlabel("Time (s)")
            bx.set_ylabel("Z-score")
    
     
    def Get_RF_matrix(roi = 'All', normalize = 0, data = file_location):
        """Gives the receptive field as a matrix by  computing the difference
        between the response and baseline for each event, for specified ROIs."""
        
        # avg_bsln = Get_event_data()[3]
        if normalize == 0:
            norm = 0 
        if normalize == 1:
            norm = 1
    
        x_axis = np.empty(0)
        y_axis = np.empty(0)
        # for i in reversed(range(int(Mode/2))):
        for i in reversed(range(int(Mode/2))):
            # x_axis = np.append(Get_event_data(roi, i)[3]-Get_event_data(roi, i)[2] - avg_bsln, x_axis)
            x_axis = np.append(Get_event_data(roi, i, norm, data = data)[3]-Get_event_data(roi, i, norm, data = data)[2], x_axis)
            # a = np.flip(a)
        for j in reversed(range(int(Mode/2), Mode)):
        # for j in reversed(range(int(Mode/2), Mode)):
            # y_axis = np.append(Get_event_data(roi, j)[3]-Get_event_data(roi, j)[2] - avg_bsln, y_axis)
            y_axis = np.append(Get_event_data(roi, j, norm, data = data)[3]-Get_event_data(roi, j, norm, data = data)[2], y_axis)
            # b = np.flip(b)
            
        RF_matrix = x_axis.reshape(int(Mode/2), 1) @ y_axis.reshape(1, int(Mode/2))
        RF_matrix = np.rot90(RF_matrix, 1)
        return RF_matrix, x_axis, y_axis
    
     
    def Plot_RF(roi = 'All', normalize = 0, data = file_location, **kwargs):
        if normalize == 0:
            RF_matrix = Get_RF_matrix(roi, 0, data = data)[0]
        if normalize == 1:
            RF_matrix = Get_RF_matrix(roi, 1, data = data)[0]
    
        if 'interpolation' in kwargs:
            interpol = kwargs['interpolation']
        else:
            interpol = None
    
        vis_ang_list_rounded = np.round(vis_ang_list, 1) #axis starts at 0
        # vis_ang_list_rounded = np.round(np.absolute(vis_ang_list_alt), 1) #axis centered on 0
        
        fig, ax1 = plt.subplots(1,1, figsize = (10, 10))
        RF_plot = ax1.imshow(RF_matrix, cmap = 'bone', interpolation = interpol)
        
        ax1.set_ylabel("Visual angle (°)", labelpad = 15)
        ax1.set_yticks(np.arange(-.5, Mode/2))
        ax1.set_yticklabels(np.flip(vis_ang_list_rounded))
        ax1.yaxis.set_label_position("right")
        ax1.yaxis.tick_right()
        
        ax1.set_xlabel("Visual angle (°)", labelpad = 15)
        ax1.set_xticks(np.arange(-.5, (Mode/2)))
        ax1.set_xticklabels((vis_ang_list_rounded))
        ax1.xaxis.set_label_position("top")
        ax1.xaxis.tick_top()
        
        ax2 = ax1.secondary_xaxis('bottom')
        ax2.set_xticks(np.arange(0, Mode/2))
        ax2.set_xticklabels(np.arange(1, round((Mode/2)) + 1))
        ax2.set_xlabel("Bar location", labelpad = 15)
        
        ax2 = ax1.secondary_yaxis('left')
        ax2.set_yticks(np.arange(0, Mode/2))
        ax2.set_yticklabels(reversed(np.arange(1, round((Mode/2)) + 1)))
        ax2.set_ylabel("Bar location", labelpad = 15)
    
        plt.grid(True, which = 'major', color = "grey")
        plt.colorbar(RF_plot, fraction = 0.04 ,pad = .175, label = "Z-score difference (baseline avg. - response avg.)")
        if roi == 'All':
            plt.suptitle("Computed receptive field for all sampled ROIs", y = .90)
        if "title" in kwargs:
            plt.suptitle(kwargs["title"], y = .90)
        else:
            plt.suptitle("Computed receptive field for ROI {}".format(roi), y = .90)
        if 'saveas' in kwargs:
            plt.savefig(r'C://Users//skrem//OneDrive//Universitet//MSc//Experimental project//Figures//Python generated//{}'.format(kwargs['saveas']), dpi = 2000,  bbox_inches='tight')
            
    """Consider this 3D RF plot too! https://stackoverflow.com/questions/44895117/colormap-for-3d-bar-plot-in-matplotlib-applied-to-every-bar
    or https://www.geeksforgeeks.org/3d-surface-plotting-in-python-using-matplotlib/ or https://stackoverflow.com/questions/38698277/plot-normal-distribution-in-3d """
    
     
    def gaus(x, a, b, c):
            # a Gaussian distribution
                return a * np.exp(-(x-b)**2/(2*c**2))
    
     
    def find_near(input_array, target):
        """Return nearest value to specified target and its index in array"""
        arr = np.asarray(input_array)
        x = target
        difference_array = np.abs(arr-x)
        index = difference_array.argmin()
        nearest = arr[index]
        nearest_loc = index
        return nearest, nearest_loc
    
     
    def RF_profile(roi = 'All', normalize = 0, plot = 1, curvefit = 1, data = file_location, test_fit = True, title = 0, **kwargs):
        """Returns a barchart of X and Y response profiles for specified ROI. Differs
        from RF_matrix_slice() in that RF_profile retrieves plot BEFORE matrix 
        multiplication and subsequent matrix slicing --> E.g. RF_profile draws on 
        raw"""
        
        if normalize == 0:
            norm = 0 
        if normalize == 1:
            norm = 1
        
        if 'example_data' in kwargs:
            x_axis = kwargs['example_data'][0]
            y_axis = kwargs['example_data'][1]
            
        else:
            x_axis = np.empty(0)
            y_axis = np.empty(0)
            
            for i in reversed(range(int(Mode/2))):
                x_axis = np.append(Get_event_data(roi, i, norm, data = data)[3]-Get_event_data(roi, i, norm, data = data)[2], x_axis)
            for j in (range(int(Mode/2), Mode)):
                y_axis = np.append(Get_event_data(roi, j, norm, data = data)[3]-Get_event_data(roi, j, norm, data = data)[2], y_axis)
        
        if plot == 1:
            plt.figure(dpi = 800)
            # plt.subplot(2, 1, 1)
            
            plt.bar(np.arange(0, Mode/2), x_axis.reshape(int(Mode/2),), width=1, label = "X axis scores")
            plt.bar(np.arange(0, Mode/2), y_axis.reshape(int(Mode/2),), width=.90, label = "Y axis scores")
            
            axx = plt.gca()
            axy = axx.secondary_xaxis('top')
            if title == 1:    
                plt.title("ROI RF response profile (X and Y axes)")
            axx.set_xlabel("Visual angle (°)")
            axx.set_ylabel("Response (Z-score difference)")
            plt.xticks(np.arange(-.5, (Mode/2)))
            axx.set_xticklabels(np.round(vis_ang_list, 1))
            
            axy.set_xticks(np.arange(0, Mode/2))
            axy.set_xticklabels(np.arange(0, round((Mode/2))))
            axy.set_xlabel("Bar position")
            
            handles, labels = axx.get_legend_handles_labels()
            plt.legend(reversed(handles), reversed(labels))
        
        if curvefit == 1: #for plotting purposes
            xdata = np.arange(0, int(Mode/2))
            x_ydata = x_axis.reshape(int(Mode/2),)
            y_ydata = y_axis.reshape(int(Mode/2),)
            
            #Get curve params 
            popt_x, pcov_x = scipy.optimize.curve_fit(gaus, xdata, x_ydata, maxfev=2500, p0 = np.array((max(x_ydata), np.argmax(x_ydata),1)), bounds = ((-np.inf, -np.inf, -np.inf), (max(x_ydata), np.inf, np.inf)))
            popt_y, pcov_y = scipy.optimize.curve_fit(gaus, xdata, y_ydata, maxfev=2500, p0 = np.array((max(y_ydata), np.argmax(y_ydata),1)), bounds = ((-np.inf, -np.inf, -np.inf), (max(y_ydata), np.inf, np.inf)))
            
            #Plot curve
            resolution = 1000
            x=np.linspace(0, Mode/2, resolution)
            yx = gaus(x, *popt_x)
            yy = gaus(x, *popt_y)
            
            if test_fit == True:
                #Compute R^2 --> https://stackoverflow.com/questions/19189362/getting-the-r-squared-value-using-curve-fit
                x_residuals = x_ydata - gaus(xdata, *popt_x) #Get residuals
                x_ss_res = np.sum(x_residuals**2) #Calculate residual sum of squares
                x_ss_tot = np.sum((x_ydata - np.mean(x_ydata))**2) #Total sum of squares
                x_r_squared = 1 - (x_ss_res / x_ss_tot) #R^2 value
                x_r = np.sqrt(x_r_squared)
                
                y_residuals = y_ydata - gaus(xdata, *popt_y) 
                y_ss_res = np.sum(y_residuals**2) 
                y_ss_tot = np.sum((y_ydata - np.mean(y_ydata))**2)
                y_r_squared = 1 - (y_ss_res / y_ss_tot)
                y_r = np.sqrt(y_r_squared)
            
                #Compute Adjusted R^2 --> https://www.statisticshowto.com/probability-and-statistics/statistics-definitions/adjusted-r2/
                regs = len(np.array(gaus.__code__.co_varnames))-1 #Number of regressors (variables in model - constant)
                
                x_n = len(x_ydata) #n number of points in data sample (of curve or data?)
                x_r_squared_adj = 1 - ((1-x_r_squared)*(x_n - 1))/(x_n-regs-1)
                
                y_n = x_n
                y_r_squared_adj = 1 - ((1-y_r_squared)*(y_n - 1))/(y_n-regs-1)
                
                if plot == 1:            
                    #Put R^2 and Chi^2 values into a little table
                    table_content = np.array([["R", np.round(x_r, 2), np.round(y_r, 2)], ["R\u00b2", np.round(x_r_squared, 2), np.round(y_r_squared, 2)],["R\u2090\u00b2", np.round(x_r_squared_adj, 2), np.round(y_r_squared_adj, 2)]]) #["X\u00b2", np.round(x_chi_p, 2), np.round(y_chi_p, 2)]]) #placeholder
                    collabel = ('Fit', 'X', 'Y')
                    The_table = plt.table(cellText=table_content ,colLabels=collabel, colWidths = [0.05]*3, loc = 'bottom left', bbox = (-.1,-.4,.25,.25))
                    The_table.scale(1 * 1.5, 1)
            
            if plot == 1:
                x_curve_eq = r"$\ f(x) = %.2f e ^ {-\frac{(x - %.2f)^2}{(%.2f)^2}} "\
                    "$" % (popt_x[0], popt_x[1], 2*popt_x[2])
                y_curve_eq = r"$\ f(y) = %.2f e ^ {-\frac{(y - %.2f)^2}{(%.2f)^2}} "\
                    "$" % (popt_y[0], popt_y[1], 2*popt_y[2])
                plt.plot(x, yx, c='b', label="{}".format(x_curve_eq),
                      path_effects=[path_effects.Stroke(linewidth=4,
                      foreground = 'black'), path_effects.Normal()])
                plt.plot(x, yy, c = 'orange', label = y_curve_eq,
                         path_effects=[path_effects.Stroke(linewidth = 4,
                       foreground = 'black'), path_effects.Normal()])
                
                plt.xticks(np.arange(-.5, (Mode/2)))
                handles, labels = axx.get_legend_handles_labels()
                plt.legend(reversed(handles), (reversed(labels)))
                axx.set_xticklabels(np.round(vis_ang_list, 1))
         
        if plot == 1:
            plt.show()
            
        if curvefit == 0:
            return x_axis, y_axis
        if curvefit == 1 and test_fit == True:
            return x_axis, y_axis, x_r_squared, y_r_squared
        else:
            return x_axis, y_axis
    
     
    def RF_matrix_slice (roi = 'All', normalize = 0, plot = 1, curvefit = 1, data = file_location):
        if normalize == 0:
            RF_matrix = Get_RF_matrix(roi, 0, data)[0]
        if normalize == 1:
            RF_matrix = Get_RF_matrix(roi, 1, data)[0]
        
        # RF_peak = np.amax(RF_matrix)
        RF_peak_loc = np.where(RF_matrix == np.amax(RF_matrix))
        
        y_axis_vals = RF_matrix[:, RF_peak_loc[1]]
        x_axis_vals = RF_matrix[RF_peak_loc[0]]
        
        if plot == 1:
            plt.figure(dpi = 800)
            
            plt.bar(np.arange(0, Mode/2), x_axis_vals.reshape(int(Mode/2),), width=1, label = "X axis scores")
            plt.bar(np.arange(0, Mode/2), y_axis_vals.reshape(int(Mode/2),), width=.90, label = "Y axis scores")
            
            axx = plt.gca()
            axy = axx.secondary_xaxis('top')
            plt.title("Slice through centre of RF matrix (X and Y axes)")
            axx.set_xticks(np.arange(0, Mode/2))
            axx.set_xlabel("Visual angle (°)")
            axx.set_ylabel("Response (Z-score difference)")
            
            axy.set_xticks(np.arange(0, Mode/2))
            axy.set_xticklabels(np.arange(0, round((Mode/2))))
            axy.set_xlabel("Bar position")
            
            handles, labels = axx.get_legend_handles_labels()
            plt.legend(reversed(handles), reversed(labels))
            
            if curvefit == 1:
                xdata = np.arange(0, int(Mode/2))
                x_ydata = x_axis_vals.reshape(int(Mode/2),)
                y_ydata = y_axis_vals.reshape(int(Mode/2),)    
            
                # popt_x, pcov_x = scipy.optimize.curve_fit(gaus, np.arange(0, int(Mode/2)), x_axis_vals.reshape(int(Mode/2),), maxfev=2500)
                # popt_y, pcov_y = scipy.optimize.curve_fit(gaus, np.arange(0, int(Mode/2)), y_axis_vals.reshape(int(Mode/2),), maxfev=2500)
                popt_x, pcov_x = scipy.optimize.curve_fit(gaus, xdata, x_ydata, maxfev=2500, p0 = np.array((max(x_ydata), np.argmax(x_ydata), 1)), bounds = ((-np.inf, -np.inf, -np.inf), (max(x_ydata), np.inf, np.inf)))
                popt_y, pcov_y = scipy.optimize.curve_fit(gaus, xdata, y_ydata, maxfev=2500, p0 = np.array((max(y_ydata), np.argmax(y_ydata), 1)), bounds = ((-np.inf, -np.inf, -np.inf), (max(y_ydata), np.inf, np.inf)))
                
                x=np.linspace(0, Mode/2, 1000)
                yx = gaus(x, *popt_x)
                yy = gaus(x, *popt_y)
                
                x_curve_eq = r"$\ f(x) = %.2f e ^ {-\frac{(x - %.2f)^2}{(%.2f)^2}} "\
                    "$" % (popt_x[0], popt_x[1], 2*popt_x[2])
                y_curve_eq = r"$\ f(y) = %.2f e ^ {-\frac{(y - %.2f)^2}{(%.2f)^2}} "\
                    "$" % (popt_y[0], popt_y[1], 2*popt_y[2])
                plt.plot(x, yx, c='b', label="{}".format(x_curve_eq),
                      path_effects=[path_effects.Stroke(linewidth=4,
                      foreground = 'black'), path_effects.Normal()])
                plt.plot(x, yy, c = 'orange', label = y_curve_eq,
                         path_effects=[path_effects.Stroke(linewidth = 4,
                       foreground = 'black'), path_effects.Normal()])
                
                plt.xticks(np.arange(-.5, (Mode/2)))
                handles, labels = axx.get_legend_handles_labels()
                plt.legend(reversed(handles), reversed(labels))
                axx.set_xticklabels(np.round(vis_ang_list, 1))
     
            plt.show()
        return x_axis_vals, y_axis_vals
    
     
    def Compute_RF_size(roi = 'All', normalize = 0, plot = 0, data = file_location, test_fit = True, **kwargs):
        """https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2269911/"""
        
        if 'example_data' in kwargs:
            x_vals = kwargs['example_data'][0]
            y_vals = kwargs['example_data'][1]
    
        else: 
            if normalize == 0:
                x_vals, y_vals = RF_profile(roi, 0, 0, data = data)[:2]
            if normalize == 1:
                x_vals, y_vals = RF_profile(roi, 1, 0, data = data)[:2]
                
        xdata = np.arange(0, int(Mode/2))
        x_ydata = x_vals.reshape(int(Mode/2),)
        y_ydata = y_vals.reshape(int(Mode/2),) 
        
        try:    
            popt_x, pcov_x = scipy.optimize.curve_fit(gaus, xdata, x_ydata, maxfev=2500, p0 = np.array((max(x_ydata), np.argmax(x_ydata), 1)), bounds = ((-np.inf, -np.inf, -np.inf), (max(x_ydata), np.inf, np.inf)))
            popt_y, pcov_y = scipy.optimize.curve_fit(gaus, xdata, y_ydata, maxfev=2500, p0 = np.array((max(y_ydata), np.argmax(y_ydata), 1)), bounds = ((-np.inf, -np.inf, -np.inf), (max(y_ydata), np.inf, np.inf)))
        
        except Exception:
            popt_x = pcov_x = popt_y = pcov_y = 0
            Nofit = 'No fit'
            print ("scipy.optimize.curve_fit maxfev reached, returned (None, None) dtype = object") 
            return Nofit, Nofit
                
        resolution = 10000 #how many points on curve, more is better but computationally slower
        index_to_visang = vis_ang_list[-1]/resolution
        x=np.linspace(0, Mode/2, resolution)
        yx = gaus(x, *popt_x)
        yy = gaus(x, *popt_y)
        
        criteria = 0.005
    
        yx_peak = np.where(yx == np.amax(yx))[0][0]
        if yx_peak == 0 or yx_peak == resolution:
            yx_half_width = "Peak obscured"#None #return None if value falls outside range of data
        yx_curve_indeces = np.where(yx > criteria)
        yx_left_index = yx_curve_indeces[0][0]
        yx_right_index = yx_curve_indeces[0][-1]
        yx_half_width = ((yx_right_index - yx_left_index) * index_to_visang) / 2
        if yx_left_index == 0 or yx_right_index == resolution:
            yx_half_width = "Half-width obscured"
    
    
        yy_peak = np.where(yy == np.amax(yy))[0][0]
        if yy_peak == 0 or yy_peak == resolution:
            yy_half_width = "Peak obscured"#None #return None if value falls outside range of data
        yy_curve_indeces = np.where(yy > criteria)
        yy_left_index = yy_curve_indeces[0][0]
        yy_right_index = yy_curve_indeces[0][-1]
        yy_half_width = ((yy_right_index - yy_left_index) * index_to_visang) / 2
        if yy_left_index == 0: #or yy_right_index == resolution:
            yy_half_width = "Half-width obscured"
                
        
        if test_fit == True:
            x_axis = np.empty(0)
            y_axis = np.empty(0)
    
            for i in reversed(range(int(Mode/2))):
                x_axis = np.append(Get_event_data(roi, i, normalize, data = data)[3]-Get_event_data(roi, i, normalize, data = data)[2], x_axis)
            for j in (range(int(Mode/2), Mode)):
                y_axis = np.append(Get_event_data(roi, j, normalize, data = data)[3]-Get_event_data(roi, j, normalize, data = data)[2], y_axis)
            
            xdata = np.arange(0, int(Mode/2))
            x_ydata = x_axis.reshape(int(Mode/2),)
            y_ydata = y_axis.reshape(int(Mode/2),)
    
            x_y = x_ydata 
            y_y = y_ydata
            X = gaus(xdata, *popt_x)
            
            x_pearsons_r = scipy.stats.pearsonr(x_y, gaus(xdata, *popt_x))
            x_r_sqrd = metrics.r2_score(x_y, gaus(xdata, *popt_x))
            x_r_squared_adjusted = 1 - ((1 - x_r_sqrd)*(len(x_y) - 1)) / ((len(x_y) - len(popt_x) - 1))
            spearmans_for_x = scipy.stats.spearmanr(x_y, gaus(xdata, *popt_x))
            
            y_pearsons_r = scipy.stats.pearsonr(y_y, gaus(xdata, *popt_y))
            y_r_sqrd = sk.metrics.r2_score(y_y, gaus(xdata, *popt_y))
            y_r_squared_adjusted = 1 - ((1 - y_r_sqrd)*(len(y_y) - 1)) / ((len(y_y) - len(popt_y) - 1))
            spearmans_for_y = scipy.stats.spearmanr(y_y, gaus(xdata, *popt_y))
            
        
        if plot == 1:
            plt.plot(np.linspace(0, vis_ang_list[-1], resolution), yx)
            plt.plot(np.linspace(0, vis_ang_list[-1], resolution), yy)
            if isinstance(yx_half_width, str) == False:
                plt.hlines(yx[int((yx_left_index + yx_peak) / 2)], yx_left_index * index_to_visang + yx_half_width/2, yx_right_index * index_to_visang - yx_half_width/2)
                plt.hlines(yx[int((yx_left_index + yx_peak) / 2)], yx_left_index * index_to_visang, yx_right_index * index_to_visang, linestyle = 'dotted', colors = 'k') 
                plt.vlines(x = yx_left_index*index_to_visang, ymin = 0, ymax = yx[int((yx_left_index + yx_peak) / 2)], linestyle = 'dotted', colors = 'k')
                plt.vlines(x = yx_left_index * index_to_visang + yx_half_width/2, ymin = 0, ymax = yx[int((yx_left_index + yx_peak) / 2)])
                plt.vlines(x = yx_right_index*index_to_visang, ymin = 0, ymax = yx[int((yx_left_index + yx_peak) / 2)], linestyle = 'dotted', colors = 'k')
                plt.vlines(x = yx_right_index * index_to_visang - yx_half_width/2, ymin = 0, ymax = yx[int((yx_left_index + yx_peak) / 2)])
            
            if isinstance(yy_half_width, str) == False:
                plt.hlines(yy[int((yy_left_index + yy_peak) / 2)], yy_left_index * index_to_visang + yy_half_width/2, yy_right_index * index_to_visang - yy_half_width/2,  colors = '#FF8317')
                plt.hlines(yy[int((yy_left_index + yy_peak) / 2)], yy_left_index * index_to_visang, yy_right_index * index_to_visang, linestyle = 'dotted', colors = 'k') 
                plt.vlines(x = yy_left_index*index_to_visang, ymin = 0, ymax = yy[int((yy_left_index + yy_peak) / 2)], linestyle = 'dotted', colors = 'k')
                plt.vlines(x = yy_left_index * index_to_visang + yy_half_width/2, ymin = 0, ymax = yy[int((yy_left_index + yy_peak) / 2)],  colors = '#FF8317')
                plt.vlines(x = yy_right_index*index_to_visang, ymin = 0, ymax = yy[int((yy_left_index + yy_peak) / 2)], linestyle = 'dotted', colors = 'k')
                plt.vlines(x = yy_right_index * index_to_visang - yy_half_width/2, ymin = 0, ymax = yy[int((yy_left_index + yy_peak) / 2)],  colors = '#FF8317')
                
            plt.axvline(x = yx_peak*index_to_visang, c = 'g', linestyle = (0, (5, 10)))
            plt.axvline(x = yy_peak*index_to_visang, c = 'g', linestyle = (0, (5, 10)))
            
            # plt.xlim(0, 75)
            plt.xlabel("Visual angle (°)")
            
            print("Pearsons X: {}, {}".format(x_pearsons_r, y_pearsons_r))
            print("R2: {} {}".format(x_r_sqrd, y_r_sqrd))
            print("R2adj {}, {}".format(x_r_squared_adjusted, y_r_squared_adjusted))
            print("Spearman R: {}, {}".format(spearmans_for_x, spearmans_for_y))
            
            if 'title' in kwargs:
                plt.title(kwargs["title"])
            plt.show()
        # return yx_RF_size, yy_RF_size
        return yx_half_width, yy_half_width
    
    def Model_RF_size(roi = 'All', normalize = 0, plot = 0, data = file_location, test_fit = True):
        """https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2269911/"""
        if normalize == 0:
            x_vals, y_vals = RF_profile(roi, 0, 0, 0, data = data)[:2]
        if normalize == 1:
            x_vals, y_vals = RF_profile(roi, 1, 0, 0, data = data)[:2]
    
        xdata = np.arange(0, int(Mode/2))
        x_ydata = x_vals.reshape(int(Mode/2),)
        y_ydata = y_vals.reshape(int(Mode/2),) 
    
        try:    
            popt_x, pcov_x = scipy.optimize.curve_fit(gaus, xdata, x_ydata, maxfev=2500, p0 = np.array((max(x_ydata), np.argmax(x_ydata), 1)), bounds = ((-np.inf, -np.inf, -np.inf), (max(x_ydata), np.inf, np.inf)))
            popt_y, pcov_y = scipy.optimize.curve_fit(gaus, xdata, y_ydata, maxfev=2500, p0 = np.array((max(y_ydata), np.argmax(y_ydata), 1)), bounds = ((-np.inf, -np.inf, -np.inf), (max(y_ydata), np.inf, np.inf)))
            
        except Exception:
            popt_x = pcov_x = popt_y = pcov_y = 0
            Nofit = 'No fit'
            print ("scipy.optimize.curve_fit maxfev reached, returned (None, None) dtype = object") 
            return Nofit, Nofit
        
        resolution = 10000 #how many points on curve, more is better but computationally slower
        x=np.linspace(-Mode, Mode, resolution)
        index_to_visang = vis_ang_list[-1]*4/resolution #multiply by 2 because 
        yx = gaus(x, *popt_x)
        yy = gaus(x, *popt_y)
        buffer_estimate = .05 #If the first index is within x percentage of half-height, count it as half-height 
        
        criteria = 0.005
    
        yx_peak = np.where(yx == np.amax(yx))[0][0]
        if yx_peak == 0 or yx_peak == resolution:
            yx_half_width = "Peak obscured"#None #return None if value falls outside range of data
        yx_curve_indeces = np.where(yx > criteria)
        yx_left_index = yx_curve_indeces[0][0]
        yx_right_index = yx_curve_indeces[0][-1]
        yx_half_width = ((yx_right_index - yx_left_index) * index_to_visang) / 2
        if yx_left_index == 0 or yx_right_index == resolution:
            yx_half_width = "Half-width obscured"
    
    
        yy_peak = np.where(yy == np.amax(yy))[0][0]
        if yy_peak == 0 or yy_peak == resolution:
            yy_half_width = "Peak obscured"#None #return None if value falls outside range of data
        yy_curve_indeces = np.where(yy > criteria)
        yy_left_index = yy_curve_indeces[0][0]
        yy_right_index = yy_curve_indeces[0][-1]
        yy_half_width = ((yy_right_index - yy_left_index) * index_to_visang) / 2
        if yy_left_index == 0: #or yy_right_index == resolution:
            yy_half_width = "Half-width obscured"
        
        if test_fit == True:
            x_axis = np.empty(0)
            y_axis = np.empty(0)
    
            for i in reversed(range(int(Mode/2))):
                x_axis = np.append(Get_event_data(roi, i, normalize, data = data)[3]-Get_event_data(roi, i, normalize, data = data)[2], x_axis)
            for j in (range(int(Mode/2), Mode)):
                y_axis = np.append(Get_event_data(roi, j, normalize, data = data)[3]-Get_event_data(roi, j, normalize, data = data)[2], y_axis)
            
            xdata = np.arange(0, int(Mode/2))
            x_ydata = x_axis.reshape(int(Mode/2),)
            y_ydata = y_axis.reshape(int(Mode/2),)
            
            
            spearmans_for_x = scipy.stats.spearmanr(x_ydata, gaus(xdata, *popt_x))
            x_r = spearmans_for_x[0]
            
            spearmans_for_y = scipy.stats.spearmanr(y_ydata, gaus(xdata, *popt_y))
            y_r = spearmans_for_y[0]
            
        if plot == 1:
            plt.plot((np.linspace(-vis_ang_list[-1]*2, vis_ang_list[-1]*2, resolution)), yx)
            plt.plot((np.linspace(-vis_ang_list[-1]*2, vis_ang_list[-1]*2, resolution)), yy)
            if isinstance(yx_half_width, str) == False:
                plt.hlines(yx[int((yx_left_index + yx_peak) / 2)], yx_left_index * index_to_visang -  vis_ang_list[-1]*2 + yx_half_width/2, yx_right_index * index_to_visang -  vis_ang_list[-1]*2 - yx_half_width/2)
                plt.hlines(yx[int((yx_left_index + yx_peak) / 2)], yx_left_index * index_to_visang -  vis_ang_list[-1]*2, yx_right_index * index_to_visang -  vis_ang_list[-1]*2, linestyle = 'dotted', colors = 'k') 
                plt.vlines(x = yx_left_index*index_to_visang -  vis_ang_list[-1]*2, ymin = 0, ymax = yx[int((yx_left_index + yx_peak) / 2)], linestyle = 'dotted', colors = 'k')
                plt.vlines(x = yx_left_index * index_to_visang -  vis_ang_list[-1]*2 + yx_half_width/2, ymin = 0, ymax = yx[int((yx_left_index + yx_peak) / 2)])
                plt.vlines(x = yx_right_index*index_to_visang -  vis_ang_list[-1]*2, ymin = 0, ymax = yx[int((yx_left_index + yx_peak) / 2)], linestyle = 'dotted', colors = 'k')
                plt.vlines(x = yx_right_index * index_to_visang -  vis_ang_list[-1]*2 - yx_half_width/2, ymin = 0, ymax = yx[int((yx_left_index + yx_peak) / 2)])
        
            if isinstance(yy_half_width, str) == False:
                plt.hlines(yy[int((yy_left_index + yy_peak) / 2)], yy_left_index * index_to_visang -  vis_ang_list[-1]*2 + yy_half_width/2, yy_right_index * index_to_visang -  vis_ang_list[-1]*2 - yy_half_width/2, colors = '#FF8317')
                plt.hlines(yy[int((yy_left_index + yy_peak) / 2)], yy_left_index * index_to_visang -  vis_ang_list[-1]*2, yy_right_index * index_to_visang -  vis_ang_list[-1]*2, linestyle = 'dotted', colors = 'k') 
                plt.vlines(x = yy_left_index*index_to_visang -  vis_ang_list[-1]*2, ymin = 0, ymax = yy[int((yy_left_index + yy_peak) / 2)], linestyle = 'dotted', colors = 'k')
                plt.vlines(x = yy_left_index * index_to_visang -  vis_ang_list[-1]*2 + yy_half_width/2, ymin = 0, ymax = yy[int((yy_left_index + yy_peak) / 2)], colors = '#FF8317')
                plt.vlines(x = yy_right_index*index_to_visang -  vis_ang_list[-1]*2, ymin = 0, ymax = yy[int((yy_left_index + yy_peak) / 2)], linestyle = 'dotted', colors = 'k')
                plt.vlines(x = yy_right_index * index_to_visang -  vis_ang_list[-1]*2 - yy_half_width/2, ymin = 0, ymax = yy[int((yy_left_index + yy_peak) / 2)], colors = '#FF8317')    
    
            plt.axvline(x = yx_peak*index_to_visang - vis_ang_list[-1]*2, c = 'g', linestyle = (0, (5, 10)))
            plt.axvline(x = yy_peak*index_to_visang - vis_ang_list[-1]*2, c = 'g', linestyle = (0, (5, 10)))
            
            plt.xlabel("Visual angle (°)")
            plt.show()
            
        # return yx_RF_size, yy_RF_size, x_r, y_r #, x_pearsons_r, y_pearsons_r
        return yx_half_width, yy_half_width, x_r, y_r
    
    def RF_estimates_list(function, stimfolder, resolutionfolder, rootfolder = 'D:\\Dissertation files\\Further analysis'):
        """Returns a list of RF estimates based on script Compute_RF_size, for each 
        condition, for each file, for each ROI."""
        
        # stim = rootfolder + '\\' + stimfolder
        res = rootfolder + '\\' + stimfolder + '\\' + resolutionfolder 
        conds = os.listdir(res)
        # All_estimates = []
        Compare_estimates = []
        Total_ROIs = 0
        Total_R_eligible = 0
        for j in conds: #Conditions to loop through
            print(j)
            txt_files = []
            dir_files = os.listdir(res + '\\' + j)
            intermediate_list = []
            for file in dir_files: #Build list of files to loop through
                if file.endswith('.txt') is True:
                    txt_files.append(file)
            for file in txt_files: #Then loop through those files
                print(file)
                file_dir = res + '\\' + j + '\\' + file
                curr_data = Avg_data_getter(file_dir)
                if file == txt_files[len(txt_files)-1]:
                    Compare_estimates.append(intermediate_list)
                for roi in curr_data[0].columns:
                    estimate = function(roi, normalize = 1, plot = 0, data = file_dir)
                    print(r"Currently on ROI#:{} RF estimate: {} ".format(Total_ROIs, estimate[:2]), flush = True, end = '')
                    Total_ROIs += 1
                    # if isinstance(estimate[2], float) and isinstance(estimate[3], float):
                    if len(estimate) > 2:
                        if estimate[2] >= 0.5 and estimate[3] >= 0.5:
                            intermediate_list.append(estimate[:2])
                            Total_R_eligible += 1
                            print("R values: {}, {}".format(estimate[2], estimate[3]))
                    # else:
                            
                        else:
                            print("R values: {}, {} REJECTED!".format(estimate[2], estimate[3]))
                    if roi == len(curr_data[0].columns)-1:
                        print(" - Number of ROIs in file = {}".format(len(curr_data[0].columns)))
                        print(" - Total number of ROIS = {}".format(Total_ROIs))
                        print(" - N ROIs with sufficient R = {}".format(Total_R_eligible))
        Compare_estimates.append(conds)
            
        return Compare_estimates
    
    
    def Discard_junk_data(data_list, conditions = 4):
        """If index contains a string or the value 0, discard those indexes and 
        return a "cleaned" list. """
    
        data_copy = copy.deepcopy(data_list)
        conds = data_list[conditions][:]
        cleaned_list = []
        for i in range(conditions):
            cleaned_list.append([])
    
        for n, i in enumerate(data_copy[0:conditions]):
            for j in data_copy[n]:
                    cleaned_list[n] = [k for k in data_copy[n] 
                                       if isinstance(k[0],str) is False 
                                       and isinstance(k[1],str) is False
                                       and k[0] != 0 and k[1] != 0]
                    
        cleaned_list.append(conds)
        return cleaned_list
    
    def Plot_ellipses(X_width, Y_width, **kwargs):
        fig = plt.figure(figsize = (5, 5), dpi = 500)
        a = X_width
        b = Y_width
        if X_width > Y_width:
            ecc = np.sqrt(X_width**2 - Y_width**2) / X_width
        if X_width < Y_width:
            ecc = np.sqrt(Y_width**2 - X_width**2) / Y_width
        if X_width == Y_width:
            ecc = np.sqrt(X_width**2 - Y_width**2) / X_width
        """ -TODO: Implement eccentricity variable so that you can specify ecc"""
        if 'ecc' in kwargs:
            ecc = kwargs['ecc']
            X_width = 1
            Y_width = 1
        t = np.linspace(0, 2*np.pi, 1000)
        x = a * np.cos(t)
        y = b * np.sin(t)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.text(.75, -.01, "Ecc = {}".format(np.round(ecc, 3)), transform = ax.transAxes)
        plt.plot(x, y)
        plt.show()
    
    
    def Compute_ellipse(X_width, Y_width, plot = 0):
        """ Computes the eccentricity, area, and perimiter of ellipse given X and Y dims.
        
        (x - c₁)² / a² + (y - c₂)² / b² = 1, where....:
            
        - (x, y) are the variables - the coordinates of an arbitrary point on the ellipse;
        - (c₁, c₂) are the coordinates of the ellipse's center;
        - a is the distance between the center and the ellipse's vertex, lying on the horizontal axis;
        - b is the distance between the center and the ellipse's vertex, lying on the vertical axis.
        
        c₁ and c₂ are assumed to be 0, 0, meaning ellipses are centered.
        
        Returns
        -------
        X_dim: Vis ang (°)
        Y_dim: Vis ang (°)
        Eccentricity: Scale from 0 = Circle, 1 = basically flat
        Area: Divided by stim_width_visang (so, mm) --> not currently true 
        """
        
        X_dim = X_width
        Y_dim = Y_width
        
        if X_width > Y_width:
            ecc = np.sqrt(X_width**2 - Y_width**2) / X_width
        if X_width < Y_width:
            ecc = np.sqrt(Y_width**2 - X_width**2) / Y_width
        if X_width == Y_width:
            ecc = np.sqrt(X_width**2 - Y_width**2) / X_width
        
        # area = np.sqrt((np.pi * X_width * Y_width)) #Area of ellipses: Area = Pi * A * B
        area = (np.pi * X_dim/2 * Y_dim/2) #Area of ellipses: Area = Pi * A * B
        # perim = np.pi * (X_width + Y_width) * (1 + 3 *(X_width - Y_width)**2 / (10 + np.sqrt((4 - 3* X_width - Y_width)**2 / (X_width + Y_width)**2))) #Ramanujan approximation
        
        if plot == 1:
            Plot_ellipses(X_width, Y_width)
            
        return X_dim, Y_dim, ecc, area #, area, perim
        
    
    def RF_ellipses_list(two_dim_RF_list, conditions = 4):
        RF_list = two_dim_RF_list
        
        ellipse_list = []
        for i in range(conditions):
            ellipse_list.append([])
        
        for n, i in enumerate(RF_list[:conditions]):
            for j in RF_list[n]:
                Ellipse_data = Compute_ellipse(j[0], j[1])
                ellipse_list[n].append(Ellipse_data)
        
        conds = two_dim_RF_list[conditions][:]
        ellipse_list.append(conds)
        
        return ellipse_list
    
       
    def List_ellipse_params(ellipse_list, conditions = 4, get_avg = 0):
        all_Xs = []
        all_Ys = []
        all_eccs = []
        all_areas = []
        
        for i in ellipse_list[:conditions]:
            cond_x = []
            cond_y = []
            cond_ecc = []
            cond_area = []
            for n, j in enumerate(i):
                cond_x.append(j[0])
                cond_y.append(j[1])
                cond_ecc.append(j[2])
                cond_area.append(j[3])
                if j == i[-1]:
                    all_Xs.append(cond_x)
                    all_Ys.append(cond_y)
                    all_eccs.append(cond_ecc)
                    all_areas.append(cond_area)
                    
        if get_avg == 1:    
            avg_Xs = np.empty((conditions,1))
            avg_Ys = np.empty((conditions,1))
            avg_eccs = np.empty((conditions,1))
            avg_areas = np.empty((conditions,1))
            
            for n, i in enumerate(all_Xs):
                avg_Xs[n] = np.average(i)
            for m, j in enumerate(all_Ys):
                avg_Ys[m] = np.average(j)
            for l, k in enumerate(all_eccs):
                avg_eccs[l] = np.average(k)
            for k, l in enumerate(all_areas):
                avg_areas[k] = np.average(l)
    
            return avg_Xs, avg_Ys, avg_eccs
        else:
            return all_Xs, all_Ys, all_eccs, all_areas
    
    def ellipse_param_dfs(RF_ellipses):
        All_Xs = List_ellipse_params(RF_ellipses)[0]
        All_Ys = List_ellipse_params(RF_ellipses)[1]
        All_eccs = List_ellipse_params(RF_ellipses)[2]
        All_areas = List_ellipse_params(RF_ellipses)[3]
    
        All_Xs_df = pd.DataFrame(All_Xs).transpose()
        All_Ys_df = pd.DataFrame(All_Ys).transpose()
        All_eccs_df = pd.DataFrame(All_eccs).transpose()
        All_areas_df = pd.DataFrame(All_areas).transpose()
    
        df_list = [All_Xs_df, All_Ys_df, All_eccs_df, All_areas_df]
    
        for i in df_list:
            i.columns = conds_list
        return All_Xs_df, All_Ys_df, All_eccs_df, All_areas_df
       
    def list_deg_maker(raw_list):
        deg_list = [str(x) + degree_sign for x in raw_list]
        return deg_list
        
    
    
    def plot_amps(normalize = 0, **kwargs):
        if normalize == 0:
            norm = 0
        if normalize == 1:
            norm = 1
    
        amp_list = []
        bsl_list = []
        for i in avg_df.columns:
            event_in_amp_list = []
            event_in_bsl_list = []
            amp_list.append(event_in_amp_list)
            bsl_list.append(event_in_bsl_list)
            for j in range(Mode):
                curr_amp = Get_event_data(i, j, normalize = norm)[2]
                curr_bsl = Get_event_data(i, j, normalize = norm)[3]
                event_in_amp_list.append(curr_amp)
                event_in_bsl_list.append(curr_bsl)
        with plt.style.context('seaborn-paper'):
            fig, ax = plt.subplots(dpi = 800)
            for i in avg_df.columns:
                base_colors = mpl.cm.get_cmap('gist_rainbow') #hsv(x) for x in range(ROI_number)] <-- legacy solution
                color_list = base_colors(np.linspace(0, 1, ROI_number))
                plt.plot(amp_list[i], color = color_list[i])
                plt.plot(bsl_list[i], color = color_list[i], linestyle = 'dotted')
                plt.scatter(np.arange(0, Mode), amp_list[i], color = color_list[i])
            if 'legend' in kwargs:
                if kwargs['legend'] == 1:
                    custom_lines = [mpl.lines.Line2D([0], [0], color='k', lw=1.5),
                      mpl.lines.Line2D([0], [0], color='k', lw=1.5, linestyle = 'dotted')]
                    ax.legend(custom_lines, ['Response', 'Baseline'],  loc='upper center', bbox_to_anchor=(0.2, -0.125),
                      fancybox=True, shadow=False, ncol = 3)       
            if 'tick_interval' and 'tick_labels' in kwargs:
                plt.xticks(kwargs['tick_interval'], kwargs['tick_labels'])
            if 'title' in kwargs:
                the_title = kwargs["title"]
                plt.suptitle(the_title)
            if 'x_label' in kwargs:
                plt.xlabel(kwargs['x_label'])
            if 'y_label' in kwargs:
                plt.ylabel(kwargs['y_label'])
            plt.grid()
        return amp_list
    
    """
    TODO
    - Figure out how to make import_data 'path' most logical... --> Put in settings_init.py? Its in there for now... 
    - Put everything in classes 
        """
        
    # class Utilities():
    
    def import_data(filename, original_formatting = 0, path = settings_init.import_path):
        
        get_data = pd.read_excel(path + '\{}.xlsx'.format(filename))
        return_data = get_data.drop(get_data.columns[0], axis = 1)
        
        if original_formatting == 1:
            data_list = []
            for i in range(return_data.shape[1]):
                curr_list = return_data[return_data.columns[i]].tolist()
                data_list.append(curr_list)
            for n, j in enumerate(data_list):
                data_list[n] = [x for x in j if np.isnan(x) == False]            
            return pd.DataFrame(data_list).T
        else:
            return pd.DataFrame(return_data)
    
    def data_to_frame():
        "For now, just making manually in Excel."
        print("Under construction. For now, generate manually (e.g. export data to excel and restructure.)")
    
    "Collection of functions for statistical testing_____________________________"
    def descriptives(data_df, group):
        check = data_df[group].dropna()
        mean = np.round(np.mean(check), 3)
        stdev = np.round(np.std(check), 3)
        min_val = np.round(min(check), 3)
        Q1 = np.round(np.quantile(check, 0.25), 3)
        median = np.round(np.quantile(check, .5), 3)
        Q3 = np.round(np.quantile(check, .75), 3)
        max_val = np.round(max(check), 3)
        n = np.round(len(check), 3)
        return mean, stdev, min_val, Q1, median, Q3, max_val, n