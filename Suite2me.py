# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 14:34:52 2022

Credits to: https://www.github.com/MouseLand/suite2p/blob/main/jupyter/run_suite2p_colab_2021.ipynb

@author: SimenLab
"""

import os
import numpy as np
import pathlib
import tifffile
import suite2p
import shutil

import Import_Igor
"""Currently hard-coded..."""
import options

# import matplotlib.pyplot as plt
# import sys
# import warnings
# import matplotlib as mpl
# import time
# import utilities
# import options_BC_testing


"""
TODO Need to crop the imaging file to the trigger channel already during the
Import_Igor phase... Maybe the class I wrote earler would be useful for this.
"""

def gen_ops(ops, db):
    output_ops = suite2p.run_s2p(ops=ops, db=db)  # Run the actual algo...
    print("Initiating suite2p.run_s2p")
    # print(len(output_ops))
    output_ops_file = np.load(pathlib.Path(output_ops['save_path']).joinpath(
        'ops.npy'), allow_pickle=True).item()
    if output_ops_file.keys() != output_ops.keys():
        raise ValueError(
            "Keys in output_ops_file is different from keys in output_ops")
    return output_ops  # , output_ops_file


def extract_singleplane(input_folder, save_dir, output_folder, crop):
    """
    Script for running Suite2p analysis on .tiffs with a single plane.
    E.g., every frame is from the same plane. The .tiffs are processed in
    sequence.

    Parameters
    ----------
    input_folder: Str or pathlib.Path object
        Folder from which Igor .smh's/.smp's are taken.
    save_dir: Str or pathlib.Path object
        Directory where outputs from Suite2p are stored.
    output_folder:
        Name of folder in save_dir algorithm should output to.
    crop: Int
        Takes a single intiger and assumes it as squared (i.e. 256 (x 256), 512 (x 512), etc.)
    ops_path: Path-like
        Path of options file to use.
    Returns
    -------
    None.
    """
    # Define some handy inner functions 
    ## Checks paths and returns True/False conditionally
    def probe_path(path, look_for):
        check_here = pathlib.Path(path)
        content = check_here.rglob(f'*/{look_for}')
        for i in content:
            if look_for in i.parts:
                target_content_present = True
                break
            else:
                target_content_present = False
            return target_content_present
    ## Algorithmically generate .tiffs and .npy (image and trigger) from Igor 
    ## binaries, then save them in target folder.
    def gen_tiffs_from_igor(directory):
        tiff_count = 0
        for file in directory.iterdir():
            if file.suffix == ".smp":
                tiff_count += 1
                file = pathlib.Path(file).resolve()
                img = Import_Igor.get_stack(file)
                img_name = file.stem
                img_arr, trigger_arr = Import_Igor.get_ch_arrays(img, crop)
                trigger_trace = Import_Igor.trigger_trace(trigger_arr) # Algorithmically get the trigger trace out of trigger channel
                # save_folder = pathlib.Path(r".\Data\data_output\{}".format(img_name)) # Bit more elegant than above
                tiff_path = final_destination.joinpath(
                    img_name).with_suffix(".tiff")
                trig_path = final_destination.joinpath(
                    img_name).with_suffix(".npy")
                tifffile.imsave(tiff_path, img_arr)
                np.save(trig_path, trigger_trace)
                del img, img_name, img_arr, trigger_arr, file
    ## Run through target folder and clean it up (establishing a file hierarchy)
    def prep_file_hierarchy(directory):
        tiff_paths = []
        trig_paths = []
        directory = pathlib.Path(directory)
        path_of_tiffs = sorted(directory.glob('*.tiff'))
        path_of_trigs = sorted(directory.glob('*.npy'))
        for tiff, trig in zip(path_of_tiffs, path_of_trigs):
            ### Step 3.1: Make folder with tiff filename
            new_single_plane_folder = directory.joinpath(
                tiff.stem)
            if new_single_plane_folder.exists() == False:
                new_single_plane_folder.mkdir()
            ### Step 3.2: Move tiff file into folder
            tiff_new_location = pathlib.Path(shutil.move(
                tiff, new_single_plane_folder))
            current_tiff_name = tiff_new_location.stem
            # current_tiff_final_location =  tiff_new_location.rename(
            #     tiff_new_location.with_stem(
            #     f"{current_tiff_name}_ch1").with_suffix(".tiff"))
            # tiff_paths.append(current_tiff_final_location)
            tiff_paths.append(current_tiff_name)
            ### Step 3.3: Move .npy file (trigger trace) into folder 
            trig = pathlib.Path(trig)
            ### Step 3.2: Move trig file into folder
            trig_new_location = pathlib.Path(shutil.move(
                trig, new_single_plane_folder)).with_suffix(".npy")
            current_trig_name = trig_new_location.stem
            # current_tiff_final_location =  tiff_new_location.rename(
            #     tiff_new_location.with_stem(
            #     f"{current_trig_name}_ch2").with_suffix(".npy"))
            # tiff_paths.append(current_tiff_final_location)
            trig_paths.append(current_trig_name)
        path_of_tiffs = sorted(directory.rglob('*.tiff'))
        path_of_trigs = sorted(directory.rglob('*.npy'))
        return path_of_tiffs, path_of_trigs
    ## Run Suite2p on each .tiff file in the file hieararchy 
    def tiff_f_extract(path_of_tiffs):
        for tiff in path_of_tiffs:
            ### Point Suite2p to the right folder for analysis
            # needs to be a dictionary with a list of path(s)
            tiff_loc = pathlib.Path(tiff).parent
            db = {'data_path': [str(tiff_loc)], }
    
            """Select ops file (this should not be hard-coded)..."""
            ops = options.ops
            # ops = options_BC_testing.ops
            # ops = np.load(ops_path, allow_pickle=True)
            
            # Step 4: Run Suite2p on this newly created folder with corresponding tiff file
            output_ops = gen_ops(ops, db)
            # ops = suite2p.registration.metrics.get_pc_metrics(output_ops)
            # output_ops = gen_ops(ops, db)
            # ops = suite2p.get_pc_metrics(ops)
    ## Check if folder already exists
    input_folder = pathlib.Path(input_folder)
    save_dir = pathlib.Path(save_dir).resolve()
    """Redundant. Can incorporate into one var:"""
    output_folder = pathlib.Path(output_folder) 
    final_destination = save_dir.joinpath(output_folder)
    print("Directory info") 
    print("- Save location:", final_destination)
    print("- Currently exists?", final_destination.exists())
    try:
        # Ideally...
        ## Simply make the directory:
        os.mkdir(final_destination.resolve())
        print(f"Target directory succesfully created: {final_destination}")
        print("Running data extraction algorithms.")
        ## Fill directory with data:
        gen_tiffs_from_igor(input_folder)
        ## Organise the file hieararchy
        tiff_paths, trig_paths = prep_file_hierarchy(final_destination)
        ## Run Suite2P on organised .tiff files
        tiff_f_extract(tiff_paths)
    except FileExistsError:
        print("Cannot create a directory when it already exists:", final_destination)
        ## Contingencies for handling pre-existing files
        if final_destination.exists() is True:
            any_check = any(final_destination.rglob('*'))
            suite2p_check = any(final_destination.rglob("suite2p"))
            print("Checking pre-existing content")
            print("- Content in directory?",  any_check)
            print("- Pre-existing Suite2p?", suite2p_check)
            ## If directory already exists but is empty, fill it with data
            if any_check is False and suite2p_check is False:
                print("Target directory is empty, running data extraction algorithms.")
                ## Fill directory with data:
                gen_tiffs_from_igor(input_folder)
                ## Organise the file hieararchy
                tiff_paths, trig_paths = prep_file_hierarchy(final_destination)
                ## Run Suite2P on organised .tiff files
                tiff_f_extract(tiff_paths)
            ## If Suite2P folders detected, abort to avoid overwriting previous analyses
            if suite2p_check is True:
                print(final_destination)
                raise Warning("Suite2p-related content identified. Exiting.")
            ## If .tiff files are present, index them 
            elif any_check is True:
               pre_existing_tiffs = sorted(final_destination.rglob('*.tiff'))
               print(".tiff file(s) already exist here. Skipping conversion.")
               tiff_f_extract(pre_existing_tiffs)
        else:
            raise Warning("Unknown error when handling files.")

    print(f"Pipeline executed. Resulting files in {final_destination}")

