# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:13:31 2022

@author: SimenLab
"""
# Environment imports:
import pathlib
import numpy as np 
import pandas as pd
import pathlib
# import sys

#Local imports:
# import Suite2me as s2m
import utilities
import quantitative
import qualitative
# import quantitative_beta as quantitative

class Experiment:
    def __init__(self, directory,**kwargs):
        # Function for identifying the folders representing each plane scanned
        def index_suite2p_planes(directory):
            """
            Parameters
            ----------
            directory : str or path
                The string or path-like (e.g. pathlib) that points to directory 
                for Suite2p

            Returns
            -------
            plane_index
                List of indeces for the planes folders
            """            
            
            plane_index = sorted(self.directory.rglob('**/plane*'))
            if not plane_index:
                raise ValueError("No Suite2p planes folders identified in directory.")
            return plane_index
        # Function for identifying relevant folders based on plane number
        def index_from_path(plane_path_list):
            """
            Args:
                plane_path_list (list of str or paths): _description_

            Returns:
            F_index (list): List of .npy files that contain F traces
            """
            path_index = []
            for plane in plane_path_list:
                new_path = plane.joinpath("F.npy")
                if new_path.exists() is True:
                    path_index.append(new_path)
                else:
                    raise(
                    Warning(f"No F.npy file at location {plane_path_list}")
                    )
            return path_index
        # Function for building the .npy files hierarchically based on file indexes
        def build_from_index(index_list):
            """build_from_index _summary_

            Parameters
            ----------
            index_list : _type_
                _description_

            Returns
            -------
            _type_
                _description_
            """
            fs_list = [np.load(f, allow_pickle=True) for f in index_list]
            return np.array(fs_list)
        # Quick function for finding a given file in all planes and sorting them
        def find_files(in_target, file_str, suffix_str):
            """find_files Simply globs files in target directory depending on file
            str and suffix str

            Parameters
            ----------
            in_target : str, path, or path-like
                The directory from which to glob
            file_str : str
                The name of the files to be globbed
            suffix_str : 
                The file extension aka suffix to look for (including '.')

            Returns
            -------
            List
                List of globbed files
            """
            return sorted(in_target.glob(f'**/{file_str}{suffix_str}'))
        
        def execute_data_construction_steps():
            # Index the folder(s) containing data from Sutie2p and then
            # make pathlib object from folder path
            self.directory = pathlib.Path(directory)
            self.folder_content = utilities.file_handling.get_content(
                self.directory)
            self.plane_paths = index_suite2p_planes(self.directory)
            # Make note of number of planes
            self.number_of_planes = len(self.plane_paths)
            # Create empty lists for building DataFrames later (list of dicts)
            list_of_dicts = []
            list_of_names = []
            # Return the immediate content of the directory (i.e. not recursively)
            directory_content = sorted(directory.glob('*'))
            # Check whether directory is the "final" dir, or a dir of sub-dirs (ignore .pickle
            # which may already be created...)
            check_dir = [i.is_file() for i in directory_content if i.suffix != ".pickle"]
            if any(check_dir):
                directory_content = [directory]
            # Loop through the expeirment folders and grab the relevant data
            print("Retrieving information from:")
            for folder in directory_content:
                ## Ignore files
                if folder.is_file() is True:
                    continue # Simply skips to next entry in directory_content
                ### NOTE: In the follow script, rglob (recursive glob) is used
                ### where files are stored in Suite2p sub-dirs several levels
                ### down. This ONLY works because 'folder' is specifically pointing
                ### to a directory containing a single Suite2p folder. Otherwise,
                ### this would cause absolute chaos (with tons of files indexed).
                print("  -  ", folder)
                folder_name = folder.name
                ## Index .tiff file
                tiff_index = sorted(folder.rglob('*.tiff'))
                ## Index and get F trace
                f_index = sorted(folder.rglob('*/plane*/F.npy'))
                fs = build_from_index(f_index)[0]
                ## Identify how many cells there are
                cell_number = fs.shape[0]
                ## Identify how many frames there are
                frame_number = fs.shape[1]
                ## Index and get trigger trace
                trig_index = sorted(folder.glob("*.npy"))
                trigs = build_from_index((trig_index))
                ## Index and get iscell
                iscell_index = sorted(folder.rglob('*/plane*/iscell.npy'))
                iscells = build_from_index(iscell_index)
                ## Index and get iscell
                stats_index = sorted(folder.rglob('*/plane*/stat.npy'))
                stats = build_from_index(stats_index)
                ## Index and get spks
                spks_index = sorted(folder.rglob('*/plane*/spks.npy'))
                spks = build_from_index(spks_index)
                ## Index and get ops
                ops_index = find_files(directory, "ops", ".npy")
                ops = build_from_index((ops_index))
                # Throw that data into a dictionary
                print("Generating info_dict")
                info_dict = {
                    "folder_name": folder_name,
                    # "plane_number": number_of_planes,
                    "f_index": f_index,
                    "fs": fs,
                    "cell_numbers" : cell_number,
                    "frame_number" : frame_number,
                    "tiff_index": tiff_index,
                    "trig_index": trig_index,
                    "trigs": trigs,
                    "iscell_index": iscell_index,
                    "iscells": iscells,
                    "stats_index": stats_index,
                    "stats": stats,
                    "spks_index": spks_index,
                    "spks": spks,
                    "ops_index": ops_index,
                    "ops": ops,
                    "notes" : str()
                }
                # Add that dictionary to the list
                list_of_dicts.append(info_dict)
                print("info_dict appended to list_of_dicts")
                # Also add the folder name to name_list for convenience
                list_of_names.append(folder_name)
            # Make a DataFrame from the list of dictionaries (each row is one experiment)
            # self.panda = pd.DataFrame.from_dict(data=list_of_dicts).transpose()
            self.panda = pd.DataFrame.from_dict(data=list_of_dicts).transpose()
            print(f"List of dictionaries with information from {self.directory}"
                    "created and placed in DataFrame under self.panda.")
            # Pass any kwargs information into dataframe for documentation.
            for i in kwargs:
                print(f"Passing {i} = {kwargs[i]} to kwargs")
                self.panda.loc[i] = kwargs[i]
            ## Build averages algorithmically if specified to do so
            if "averages" in kwargs and kwargs["averages"] is True:
                print("Initiating algorithmic averaging")
                if "mode" not in kwargs:
                    raise ValueError("Missing **kwargs value 'mode', which determines"
                                    "averaging of f-traces. Please specify int value.")
                ## Initiate cells to overwrite information to (appends '0' to
                # all rows of experiment-columns)
                self.panda.loc["f_avgs"] = 0
                self.panda.loc["f_trials"] = 0
                self.panda.loc["trig_trials"] = 0
                self.panda.loc["trig_avgs"] = 0
                ## Get information from each experiment
                for i, (f, trg, md) in enumerate(
                                    zip(self.panda.loc["fs"], 
                                    self.panda.loc["trigs"], 
                                    self.panda.loc["mode"])):
                    ## Pass it to the averaging function
                    print("Averaging for experiment", self.panda.loc["folder_name"][i])
                    f_avg, f_trial, trig_trial, trig_avg = quantitative.average_signal(f, trg, md)
                    ## Insert the returned information into correct indeces
                    self.panda.loc["f_avgs"][i] = f_avg
                    self.panda.loc["f_trials"][i] = f_trial
                    self.panda.loc["trig_trials"][i] = trig_trial
                    self.panda.loc["trig_avgs"][i] = trig_avg
                self.avgs = self.panda.loc["f_avgs"]
                self.avg_trigs = self.panda.loc["trig_avgs"]
            # Make each row accessible via object (i.e. "shortcuts")
            self.names = self.panda.loc["folder_name"]
            self.trigs = self.panda.loc["trigs"]
            self.fs = self.panda.loc["fs"] # 3 dims: Plane, cell, time
            self.frames = self.panda.loc["frame_number"]
            self.ops = self.panda.loc["ops"]
            self.stats = self.panda.loc["stats"]
            self.iscells = self.panda.loc["iscells"]
            self.spks = self.panda.loc["spks"]
            self.cell_numbers = self.panda.loc["cell_numbers"]
            self.notes = self.panda.loc["notes"]
            print("Shortcuts created (e.g. self.fs for self.panda.loc['fs']")
            # ## This just makes a dictionary, which may be pointless...
            # self.dict = self.__dict__
        
        # Generate the information for pandas DataFrame based on class input
        execute_data_construction_steps()
        print("Experiment object generated")
    
        """Functions to be...:"""
        # Corerct for temporal misalignment 
        def get_cell_positions(stats):
            """
            Run through the iscell.npy file for an experiment and index each cell position.

            Parameters
            ----------
            iscell : .npy 
                The iscell.npy file resulting from running Suite2p

            Returns
            -------
            cell_positions : numpy 2d-array
                Two-dimensional array containing (X, Y) and numerical cell-index  

            """
             
            stats[i][j][k]["ypix"]
            stats[i][j][k]["xpix"]
            # Run through the numpy array, then the dictionary contained within, and list cell positions
            # return cell_positions
        
           # return position_tuple 
        def f_temporal_alignmnet(cell_positions, f_trace):
            """
            Temporally align cell responses by 

            Parameters
            ----------
            cell_positions : TYPE
                DESCRIPTION.
            f_trace : TYPE
                DESCRIPTION.

            Returns
            -------
            corrected_f_traces : TYPE
                DESCRIPTION.

            """