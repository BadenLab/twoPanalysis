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
import warnings
import time

import Import_Igor
import utilities
"""Currently hard-coded..."""
import options

# import matplotlib.pyplot as plt
# import sys

# import matplotlib as mpl

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


def extract_singleplane(input_folder, output_folder, crop, **kwargs):
    """
    Script for running Suite2p analysis on .tiffs with a single plane.
    E.g., every frame is from the same plane. The .tiffs are processed in
    sequence.

    Parameters
    ----------
    input_folder: Str or pathlib.Path object
        Folder from which Igor .smh's/.smp's are taken.
    save_dir (depricated): Str or pathlib.Path object
        Directory where outputs from Suite2p are stored.
    output_folder:
        Path where algorithm should output to.
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
    
    ## binaries, then save them in target folder.
    def copy_preexisting_tiffs():
        print("Identified .tiff files: Copying to output directory.")
        tiff_paths = list(pathlib.Path(input_folder).glob('*.tiff'))
        trig_paths = list(pathlib.Path(input_folder).glob('*.npy'))
        ### Copy over tiff files
        for input_file in tiff_paths:    
            shutil.copy2(input_file, output_folder.joinpath(input_file.name))
        ### Copy over npy files 
        for input_file in trig_paths:    
            shutil.copy2(input_file, output_folder.joinpath(input_file.name))
    ## Run through target folder and clean it up (establishing a file hierarchy)

    ## Run Suite2p on each .tiff file in the file hieararchy 
    def tiff_f_extract(path_of_tiffs):
        tiff_num = len(path_of_tiffs)
        if tiff_num == 0:
            raise Warning("No .tiff files detected by tiff_f_extract().")
        else:
            print(f"Indexed {tiff_num} .tiff files. Running Suite2P API sequentially.")
        for tiff in path_of_tiffs:
            ### Point Suite2p to the right folder for analysis
            # needs to be a dictionary with a list of path(s)
            tiff_loc = pathlib.Path(tiff).parent
            db = {'data_path': [str(tiff_loc)], }
    
            """Select ops file (this should not be hard-coded)..."""
            if ops = None:
                
            ops = options.ops
            # ops = options_BC_testing.ops
            # ops = np.load(ops_path, allow_pickle=True)
            
            # Step 4: Run Suite2p on this newly created folder with corresponding tiff file
            output_ops = gen_ops(ops, db)
            # ops = suite2p.registration.metrics.get_pc_metrics(output_ops)
            # output_ops = gen_ops(ops, db)
            # ops = suite2p.get_pc_metrics(ops)
    def select_data_extraction_type(input_folder):
        for file in sorted(input_folder.rglob('*')):
            suffix = file.suffix
            if suffix == ".tiff" or suffix == ".tif":
                copy_preexisting_tiffs()
                break
            if file.suffix == ".smp" or file.suffix == ".smh":
                # print("Igor files identified. Initiating relevant function...")
                utilities.data.gen_tiffs_from_igor(input_folder, output_folder, crop)
                break
            else:
                raise TypeError("Filetype not specified. Please select with filetype = ...")
    ## Check if folder already exists
    input_folder = pathlib.Path(input_folder)
    # save_dir = pathlib.Path(save_dir).resolve()
    output_folder = pathlib.Path(output_folder) 
    # final_destination = save_dir.joinpath(output_folder)
    print("Directory info") 
    print("- Save location:", output_folder)
    print("- Currently exists?", output_folder.exists())
    try:
        # Ideally...
        ## Simply make the directory:
        os.mkdir(output_folder.resolve())
        print(f"Target directory succesfully created: {output_folder}")
        print("Running data extraction algorithms.")
        ## Fill directory with data:
        select_data_extraction_type(input_folder)    
        ## Organise the file hieararchy
        tiff_paths, trig_paths = utilities.file_handling.prep_file_hierarchy(output_folder)
        ## Run Suite2P on organised .tiff files
        tiff_f_extract(tiff_paths)
    except FileExistsError:
        print("Cannot create a directory when it already exists:", output_folder)
        ## Contingencies for handling pre-existing files
        if output_folder.exists() is True:
            any_check = any(output_folder.rglob('*'))
            suite2p_check = any(output_folder.rglob("suite2p"))
            print("Checking pre-existing content")
            print("- Content in directory?",  any_check)
            print("- Pre-existing Suite2p?", suite2p_check)
            ## If directory already exists but is empty, fill it with data
            if any_check is False and suite2p_check is False:
                print("Target directory is empty, running data extraction algorithms.")
                ## Fill directory with data:
                select_data_extraction_type(input_folder)
                ## Organise the file hieararchy
                tiff_paths, trig_paths = utilities.file_handling.prep_file_hierarchy(output_folder)
                ## Run Suite2P on organised .tiff files
                tiff_f_extract(tiff_paths)
            ## If Suite2P folders detected, abort to avoid overwriting previous analyses
            if suite2p_check is True:
                print(output_folder)
                raise Warning("Suite2p-related content identified. Exiting.")
            ## If .tiff files are present, index them 
            elif any_check is True:
               pre_existing_tiffs = sorted(output_folder.rglob('*.tiff'))
               print(".tiff file(s) already exist here. Skipping conversion.")
               ## Organise the file hieararchy
               tiff_paths, trig_paths = utilities.file_handling.prep_file_hierarchy(output_folder)
               time.sleep(2)
               ## Run Suite2P on organised .tiff files
               tiff_f_extract(tiff_paths)
        else:
            raise Warning("Unknown error when handling files.")

    print(f"Pipeline executed. Resulting files in {output_folder}")

