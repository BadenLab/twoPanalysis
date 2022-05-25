# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 11:00:15 2022

@author: SimenLab
"""

import pathlib
import numpy as np
import tifffile
import warnings
import shutil
import pickle
import os
import Import_Igor

class file_handling:
    def load_experiment(f_path, trigger_path):
        """Loads up the files for F trace and trigger trace, taking the path for each respectively as input"    

        Args:
            f_path (str): Path in str of F trace numpy file 
            trigger_path (str): Path in str of trigger trace numpy file 

        Returns:
            f, trigger: The F and trigger as numpy arrays
        """        
        f = np.load(f_path, allow_pickle = True)
        trigger = np.load(trigger_path, allow_pickle = True)
        return f, trigger
    
    # File-handling related
    def get_Pathlib(path):
        pathlib_path = pathlib.Path(path)
        return pathlib_path
    
    def get_content(folder_path):
        """
        Takes a folder path (in str or path-like) and returns the 
    
        Parameters
        ----------
        folder_path : TYPE
            DESCRIPTION.
    
        Returns
        -------
        folder_contents : TYPE
            DESCRIPTION.
    
        """
        folder = pathlib.Path(folder_path)
        folder_contents = list()
        for child in folder.iterdir():
            # print(child)
            folder_contents.append(child)
        # content = folder.iterdir()
        return folder_contents
    
    def get_ops(path):
        ops =  np.load(path, allow_pickle=True)
        ops = ops.item()
        
    def read_item(file_path):
        """
        Utility function for quickly and correctly importing complexnumpy items
        from Suite2p folders (iscell, stats, etc.).
    
        Parameters
        ----------
        file_path : TYPE
            DESCRIPTION.
    
        Returns
        -------
        ops : TYPE
            DESCRIPTION.
    
        """
        ops = np.load(file_path, allow_pickles=True)
        ops = ops.item()
        return ops
    def prep_file_hierarchy(directory):
        tiff_paths = []
        trig_paths = []
        directory = pathlib.Path(directory)
        path_of_tiffs = sorted(directory.glob('*.tiff'))
        path_of_trigs = sorted(directory.glob('*.npy'))
        if len(path_of_trigs) == 0 and len(path_of_tiffs) == 0:
            print("No files found. Exiting.")
            return
        if len(path_of_trigs) == 0:
            print(path_of_trigs)
            warnings.warn("No .npy file detected. No trigger channel generated.")
            for tiff in path_of_tiffs:
                ### Step 3.1: Make folder with tiff filename
                new_single_plane_folder = directory.joinpath(
                    tiff.stem)
                if new_single_plane_folder.exists() is False:
                    new_single_plane_folder.mkdir()
                ### Step 3.2: Move tiff file into folder
                tiff_new_location = pathlib.Path(shutil.move(
                    tiff, new_single_plane_folder))
                current_tiff_name = tiff_new_location.stem
                tiff_paths.append(current_tiff_name)
        else:
            for tiff, trig in zip(path_of_tiffs, path_of_trigs):
                ### Step 3.1: Make folder with tiff filename
                new_single_plane_folder = directory.joinpath(
                    tiff.stem)
                if new_single_plane_folder.exists() is False:
                    new_single_plane_folder.mkdir()
                ### Step 3.2: Move tiff file into folder
                tiff_new_location = pathlib.Path(shutil.move(
                    tiff, new_single_plane_folder))
                current_tiff_name = tiff_new_location.stem
                tiff_paths.append(current_tiff_name)
                ### Step 3.3: Move .npy file (trigger trace) into folder 
                trig = pathlib.Path(trig)
                ### Step 3.2: Move trig file into folder
                trig_new_location = pathlib.Path(shutil.move(
                    trig, new_single_plane_folder)).with_suffix(".npy")
                current_trig_name = trig_new_location.stem
                trig_paths.append(current_trig_name)
        path_of_tiffs = sorted(directory.rglob('*.tiff'))
        path_of_trigs = sorted(directory.rglob('*.npy'))
        return path_of_tiffs, path_of_trigs
    
class data:
    ## Algorithmically generate .tiffs and .npy (image and trigger) from Igor 
    def gen_tiffs_from_igor(input_folder, output_folder, crop, **kwargs):
        input_folder = pathlib.Path(input_folder)
        output_folder = pathlib.Path(output_folder)
        img_count = 0
        ## Ensuren no dataloss by skipping conversion where conversion has 
        ## already taken place 
        pre_existing_content = sorted(output_folder.rglob('*'))
        pre_existing_content_names = []
        for i in pre_existing_content:
            pre_existing_content_names.append(i.stem)
        for file in input_folder.glob('*.smp'):
            if file.with_suffix(".smh").exists() is False:
                raise FileNotFoundError(f"Could not find accompanying header (.smh) file for {file} in {input_folder}.")
            img_count += 1
            if file.stem in pre_existing_content_names:
                warnings.warn("Input files and output files have the same name, skipping conversion. Please manually delete files in output folder to force conversion.")
                continue
            else:
                file = pathlib.Path(file).resolve()
                img = Import_Igor.get_stack(file)
                img_name = file.stem
                img_arr, trigger_arr = img.ch_arrays(crop)
                ## Select precision of trigger trace to 
                ## algorithmically get the trigger trace out of trigger channel
                if "trigger_precision" in kwargs:
                    if kwargs["trigger_precision"] == "line":
                        trigger_trace = img.trigger_trace_line()
                    if kwargs["trigger_precision"] == "frame":
                        trigger_trace = img.trigger_trace_frame()
                # Default to line-wise precision
                else:
                    trigger_trace = img.trigger_trace_line()
                # save_folder = pathlib.Path(r".\Data\data_output\{}".format(img_name)) # Bit more elegant than above
                tiff_path = output_folder.joinpath(
                    img_name).with_suffix(".tiff")
                trig_path = output_folder.joinpath(
                    img_name).with_suffix(".npy")
                try:
                    tifffile.imwrite(tiff_path, img_arr)
                except FileNotFoundError:
                    os.mkdir(tiff_path.parent)
                np.save(trig_path, trigger_trace)
                if "pickle" in kwargs and kwargs["pickle"] is True:
                    img_obj_path = output_folder.joinpath(
                        "TEMP_pickle").with_suffix(".pickle")
                    with open(img_obj_path, 'wb') as f:
                        pickle.dump(img, f)
                del img, img_name, img_arr, trigger_arr, file
            if img_count == 0:
                raise TypeError("No Igor .smh or .smp files were identified!")


    def crop(f, trigger, start_buffer, end_buffer):
        """
        Crop the 
    
        Parameters
        ----------
        f : TYPE
            DESCRIPTION.
        trigger : TYPE
            DESCRIPTION.
        start_buffer : TYPE
            DESCRIPTION.
        end_buffer : TYPE
            DESCRIPTION.
        tau : int
            The time interval, aka frames per seconds.
            
        Returns
        -------
        cropped_f : 2d-array
            Cropped flouresence traces
        cropped_trigger : 1d-array
            Cropped trigger signal
    
        """
        f_len               = f.shape[0]
        # seconds_to_frames   = 1/tau
        f_cropped           = f[start_buffer:f_len-end_buffer]
        trigger_cropped     = trigger[start_buffer:f_len-end_buffer]
        return f_cropped, trigger_cropped
    
    def interpolate(input_array, output_trace_resolution):
        """
        Interpolate
        
        Parameters
        ----------
        input_array : TYPE
            DESCRIPTION.
        output_trace_resolution : TYPE
            DESCRIPTION.

        Returns
        -------
        interpolated_trace : nd-array
            DESCRIPTION.

        """
        if input_array.ndim > 1 == True:
            if input_array.ndim == 2:
                interp_list = np.empty((len(input_array), output_trace_resolution))
                for n, trace in enumerate(input_array):
                    x = np.arange(0, len(trace))
                    y = trace
                    x_new = np.linspace(0, len(trace), output_trace_resolution)
                    interpolated_trace = np.interp(x_new, x, y)
                    
                    interp_list[n] = interpolated_trace
            if input_array.ndim == 3:
                interp_list = np.empty((input_array.shape[0], input_array.shape[1], output_trace_resolution))
                for n, array in enumerate(input_array):
                    for m, trace in enumerate(array):
                        x = np.arange(0, len(trace))
                        y = trace
                        x_new = np.linspace(0, len(trace), output_trace_resolution)
                        interpolated_trace = np.interp(x_new, x, y)
                        interp_list[n][m] = interpolated_trace
        
            # else:
            #     interp_list = np.empty((input_array.ndim, input_array.shape[1], output_trace_resolution))
            #     # print(input_array.shape)
            #     for n, array in enumerate(input_array):
            #         for m, trace in enumerate(array):
            #             x = np.arange(0, len(trace))
            #             y = trace
                        
            #             x_new = np.linspace(0, len(trace), output_trace_resolution)
            #             interpolated_trace = np.interp(x_new, x, y)
            #             # print(n, m)
            #             interp_list[n][m] = interpolated_trace
            #         # np.append(interpolated_trace, interp_list)
            
            
            return interp_list
        else:
            x = np.arange(0, len(input_array))
            y = input_array
            
            x_new = np.linspace(0, len(input_array), output_trace_resolution)
            interpolated_trace = np.interp(x_new, x, y)
        
            return interpolated_trace
    # Make trigger_trace
    def trigger_trace_by_frame(trigger_arr):
        raise DeprecationWarning("Calling trigger trace from Import_Igor is depricated. Instead, please call attribute trigger_trace_frame from Import_Igor.get_stack obj instead. This may change in the future.")
        #Binarise the trigger channel using frame-wise percision (fast but impercise)
        ## Make an array of appropriate dimension
        trigger_trace_arr = np.zeros((1, trigger_arr.shape[0]))[0]
        ## Loop through trigger image array
        for frame in range(trigger_arr.shape[0]):
            if np.any(trigger_arr[frame] > 1):
                trigger_trace_arr[frame] = 1
            #If trigger is in two consecutive frames, just use the first one so counting is correct
            if trigger_trace_arr[frame] == 1 and trigger_trace_arr[frame-1] == 1: 
                trigger_trace_arr[frame] = 0
        return trigger_trace_arr
    def trigger_trace_by_line(trigger_arr):
        raise DeprecationWarning("Calling trigger trace from Import_Igor is depricated. Instead, please call attribute trigger_trace_line from Import_Igor.get_stack obj instead. This may change in the future.")
        #Binarise the trigger channel using line-wise percision (slow but guaranteed percision)
        ## Make empty array that has dims frame_number x frame_size (for serialising each frame)
        trigger_trace_arr = np.empty((len(trigger_arr), trigger_arr[0].size))
        ## Loop through the input trigger array and serialise each frame
        for n, frame in enumerate(trigger_arr):
            serial = frame.reshape(1, frame.size)
            ## Place that serialised data in its correct index
            trigger_trace_arr[n] = serial
        ## Our matrix is now an array of vectors containing serialised information from each frame
        ## Reshape this matrix into one long array (pixel x pixel-value)
        serial_trigger_trace = trigger_trace_arr.reshape(1, trigger_arr.size)
        ## Then we binarise the serialised trigger data
        binarised_trigger_trace = np.where(serial_trigger_trace > 10000, 1, 0)[0]
        ## Boolean search for whether index n > n-1 (basically rising flank detection) 
        trig_onset_serial = binarised_trigger_trace[:-1] > binarised_trigger_trace[1:]
        ## Get the frame indeces for trigger onset
        trig_onset_index = np.where(trig_onset_serial > 0)
        ## Then divide each number in trig_onset_index by the amount of lines
        trigg_arr_shape = trigger_arr.shape
        lines_in_scan = trigg_arr_shape[1] * trigg_arr_shape[2]
        frame_of_trig = np.around(trig_onset_index[0]/lines_in_scan, 0)
        ## Convert back to frames 
        frame_number = len(trigger_arr)
        trig_trace = np.zeros(frame_number)
        for i in frame_of_trig:
            trig_trace[int(i)] = 1
        return trig_trace
    
# test2 = np.load(r"C:\Users\SimenLab\OneDrive - University of Sussex\Desktop\test.npy")
# v = interpolate(test2, 300)