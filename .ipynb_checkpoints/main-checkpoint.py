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
        
        def execute_required_steps():
            # Index the folder(s) containing data from Sutie2p
            # Make pathlib object from folder path
            self.directory = pathlib.Path(directory)
            self.folder_content = utilities.file_handling.get_content(
                self.directory)
            self.plane_paths = index_suite2p_planes(self.directory)
            self.number_of_planes = len(self.plane_paths)
            # Create an empty list
            list_of_dicts = []
            list_of_names = []
            # Return the immediate content of the directory (i.e. not recursively)
            directory_content = sorted(directory.glob('*'))
            # Check whether directory is the "final" dir, or a dir of sub-dirs
            check_dir = []
            for i in directory_content:
                check_dir.append(i.is_file())
            if any(check_dir) is True:
                _is_dir_of_dirs = False
                directory_content = directory.parent.glob('*')
            # else:
                # _is_dir_of_dirs = True
            # print(_is_dir_of_dirs)
            # average_signal(obj.fs[1][0].astype(float), obj.trigs[1][0].astype(float), 10)
            
            # Loop through the expeirment folders and grab the relevant data
            print("Retrieving information from:")
            # if _is_dir_of_dirs is True: # Loop through each folder

            for folder in directory_content:
                print("  -  ", folder)
                folder_name = folder.name
                ## Index .tiff file
                tiff_index = sorted(folder.rglob('*.tiff'))
                # Index and get F trace
                f_index = sorted(folder.rglob('*/plane*/F.npy'))
                # print(f_index)
                """
                TODO
                - Make multi-plane handling possible here...:
                """
                fs = build_from_index(f_index)[0]
                # Identify how mnay cells there are
                cell_number = fs.shape[0]
                ## Index and get trigger trace
                trig_index = sorted(folder.glob("*.npy"))
                trigs = build_from_index((trig_index))[0]
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
            # if _is_dir_of_dirs is False: # Just grab the folder contents directly
            #     folder = directory
            #     print("  -  ", folder)
            #     folder_name = folder.name
            #     ## Index .tiff file
            #     tiff_index = sorted(folder.rglob('*.tiff'))
            #     # Index and get F trace
            #     f_index = sorted(folder.rglob('*/plane*/F.npy'))
            #     # print(f_index)
            #     """
            #     TODO
            #     - Make multi-plane handling possible here...:
            #     """
            #     fs = build_from_index(f_index)[0]
            #     # Identify how mnay cells there are
            #     cell_number = fs.shape[0]
            #     ## Index and get trigger trace
            #     trig_index = sorted(folder.glob("*.npy"))
            #     trigs = build_from_index((trig_index))[0]
            #     ## Index and get iscell
            #     iscell_index = sorted(folder.rglob('*/plane*/iscell.npy'))
            #     # iscell_index = find_files(directory, "iscell", ".npy")
            #     iscells = build_from_index(iscell_index)
            #     ## Index and get stat
            #     stats_index = sorted(folder.rglob('*/plane*/stat.npy'))
            #     stats = build_from_index(stats_index)
            #     ## Index and get spks
            #     spks_index = sorted(folder.rglob('*/plane*/spks.npy'))
            #     spks = build_from_index(spks_index)
            #     ## Index and get ops
            #     ops_index = find_files(directory, "ops", ".npy")
            #     ops = build_from_index((ops_index))
                    
                # For trouble shooting:
                # if any((fs.shape[1], iscells.shape[1], stats.shape[1])) != cell_number:
                #     print(cell_number, "cells. Other shapes:", 
                #           "fs:", fs.shape,
                #           "iscells", iscells.shape,
                #           "stats", stats.shape)
                #     print("Irregularities in file structures detected."
                #           " Check shapes of files upon import.")
                    # sys.exit()
                
            # Throw that data into a dictionary
            info_dict = {
                "folder_name": folder_name,
                # "plane_number": number_of_planes,
                "f_index": f_index,
                "fs": fs,
                "cell_numbers" : cell_number,
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
            list_of_names.append(folder_name)
            print(f"List of dictionaries with information from {self.directory} "
                  "created and placed in DataFrame under self.panda.")
            # Make a DataFrame from the list of dictionaries (each row is one experiment)
            # self.panda = pd.DataFrame.from_dict(data=list_of_dicts).transpose()
            self.panda = pd.DataFrame.from_dict(data=list_of_dicts).transpose()
            # Make each row accessible via object
            self.names = self.panda.loc["folder_name"]
            self.trigs = self.panda.loc["trigs"]
            self.fs = self.panda.loc["fs"] # 3 dims: Plane, cell, time
            self.ops = self.panda.loc["ops"]
            self.stats = self.panda.loc["stats"]
            self.iscells = self.panda.loc["iscells"]
            self.spks = self.panda.loc["spks"]
            self.cell_numbers = self.panda.loc["cell_numbers"]
            self.notes = self.panda.loc["notes"]
            return self.panda 
        
        ## Functions
            
            # Upscale F trace temporarily to align
            
            # Per frame in movie...
            ## Per line in frame...
            ### Remove x amount of time based on cell position
                # - If a cell is earlier in scan (upper lines) 

            # Downscale corrected F trace to original temporal precision (e.g number of frames)
            
            # return corrected_f_traces
              
        # self.panda = pd.DataFrame.from_dict(self.__dict__, orient='index')
        self.dict = self.__dict__
        
        # Generate the basic information for pandas DataFrame
        execute_required_steps()
        
        ## Pass any other information to kwargs.
        for i in kwargs:
            self.panda.loc[i] = kwargs[i]
        ## If averages are mentioned, build averages algorithmically    
        if "averages" in kwargs and kwargs["averages"] is True:
            if "mode" not in kwargs:
                raise ValueError("Missing **kwargs value 'mode', which determines"
                                  "averaging of f-traces.")
            else:
                # self.panda.loc["f_avgs"] = None.astype('O')
                # self.panda.loc["f_trials"] = None.astype('O')
                # self.panda.loc["trig_trials"] = None.astype('O')
                ## Initiate cells to overwrite information to
                self.panda.loc["f_avgs"] = 0
                self.panda.loc["f_trials"] = 0
                self.panda.loc["trig_trials"] = 0
                self.panda.loc["trig_avgs"] = 0
                for i, (f, trg, md) in enumerate(
                                    zip(self.panda.loc["fs"], 
                                      self.panda.loc["trigs"], 
                                      self.panda.loc["mode"])):
                    # if trg.size != 0:
                    f_avg, f_trial, trig_trial, trig_avg = quantitative.average_signal(f, trg, md)
                    # else:
                        # print(f"Warning: experiment number {i} had no trigger channel. Skipping.")
                        # continue
                    
                    self.panda.loc["f_avgs"][i] = f_avg
                    self.panda.loc["f_trials"][i] = f_trial
                    self.panda.loc["trig_trials"][i] = trig_trial
                    self.panda.loc["trig_avgs"][i] = trig_avg
                
                    # self.panda.loc["trig_trials"] =
        # f_avg, f_trial, trig_trial = quantitative.average_signal(obj.fs[1][0], obj.trigs[1], 30)
        print("Experiment object generated")
    
    
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
    
        ### Old method:
        # self.folder_name = self.directory.name  # (?)
        # # Index F file(s)
        # self.f_index = index_from_path(self.plane_paths)
        # self.fs = build_from_index(self.f_index)
        # # Index .tiff files
        # self.tiff_index = find_files(self.directory, "*_ch1", ".tiff")
        # # Index and get trigger trace 
        # # self.trig_index = find_files(self.directory, "*_ch2", ".npy")
        # self.trig_index = sorted(self.directory.glob("*.npy"))
        # self.trigs = build_from_index((self.trig_index))
        # # Index and get iscell
        # self.iscell_index = find_files(self.directory, "iscell", ".npy")
        # self.iscells = build_from_index(self.iscell_index)
        # # Index and get iscell
        # self.stat_index = find_files(self.directory, "stat", ".npy")
        # self.stats = build_from_index(self.stat_index)
        # # Index and get ops
        # self.ops_index = find_files(self.directory, "ops", ".npy")
        # self.ops = build_from_index((self.ops_index))

# alt_location = pathlib.Path(r"D:\data_output\TEST_JUPYTER")
# bad_loc = pathlib.Path(r"D:\data_output\TEST_JUPYTER\nat_mono_2Hz")
# location = pathlib.Path(r"D:\data_output\TEST_JUPYTER\steps_lowpower_3s_steps")
# known_working_loc = pathlib.Path(r"D:\data_output\Testing\test_BC_out_AAA")
# AAAAA_loc = pathlib.Path("D:\data_output\AAAA")
# laptop = pathlib.Path(r"C:\Users\Simen\OneDrive\Universitet\PhD\test_BC_out\BC_natstim_monoUV_135")

# obj = Experiment(laptop, averages = True, mode = 5)
# pand = obj.panda

# f_avgs = pand.loc["f_avgs"]
# f_trials = pand.loc["f_trials"]
# trig_trials =  pand.loc["trig_trials"]
# trig_avg = pand.loc["trig_avgs"]
# # zz = pand.loc["f_avgs"]

# experiment = 0
# roi = 9
# # # %matplotlib widget
# qualitative.plot_averages(pand.loc["f_avgs"][experiment], pand.loc["f_trials"][experiment], pand.loc["trig_trials"][experiment], pand.loc["trig_avgs"][experiment], roi)