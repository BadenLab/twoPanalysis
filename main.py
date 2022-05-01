# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:13:31 2022

@author: SimenLab
"""
# Environment imports:
import pathlib
import numpy as np 
import pandas as pd

#Local imports:
import Suite2me as s2m
import utilities

"""
TODO
Change where trig trace and tiff are stored (into the same folder as 
'suite2p' folder)
"""

# params = Experiment.Stim_parameters(
#     Mode=20,
#     Repeats=3,
#     On_dur=4,
#     Off_dur=3,
#     Stim_width_um=100,
#     Cycle_len_s=7.0787,
#     LineDuration=0.001956
# ) # This does very little as of now

# fs, trig_trace = utilities.file_handling.load_experiment(r"C:\Users\Simen\OneDrive\Universitet\PhD\test_tiffs_environment\mono_noUV_Rtect+20um\suite2p\plane0\F.npy", 
#                                            r"C:\Users\Simen\OneDrive\Universitet\PhD\test_tiffs_environment\mono_noUV_Rtect+20um.npy")


# fs, trig_trace = utilities.file_handling.load_experiment(r"D:/data_output/test_smh_environment_2/mono_noUV_Rtect+20um/suite2p/plane0/F.npy", 
                                            # r"D:\data_output\test_smh_environment_2\mono_noUV_Rtect+20um.npy")

fs, trig_trace = utilities.file_handling.load_experiment(r"C:\Users\Simen\OneDrive\Universitet\PhD\test_BC_out\BC_natstim_monoUV_135\suite2p\plane0\F.npy",
                                            r"C:\Users\Simen\OneDrive\Universitet\PhD\test_BC_out\BC_natstim_monoUV_135\BC_natstim_monoUV_135_ch2.npy")

outpt = "test_BC_out"
inpt = "test_BCs"

# # Work
# ops = s2m.extract_singleplane(r"D:\data_input\{}".format(inpt),
#                             r"D:\data_output",
#                             "{}".format(outpt),
#                             256)

"""Work in progress:"""
# the idea with this is to load an experiment object which contains all the
# relevant information, including the extracted ROIs

# Make it so that this class returns a pandas dataframe via a dictionary or something similar 
class experiment:
    def __init__(self, folder_path):
        # Function for identifying the folders representing each plane scanned
        def index_suite2p_planes(folder_path):
            """
            Parameters
            ----------
            folder_path : str or path
                The string or path-like (e.g. pathlib) that points to directory 
                for Suite2p

            Returns
            -------
            plane_index
                List of indeces for the planes folders
            """            
            for subfolder in self.folder_path.glob('**'):
                if subfolder.name == "suite2p":
                    plane_num = 0
                    plane_index = []
                    for plane in subfolder.iterdir():
                        if plane.name[:5] == "plane":
                            plane_index.append(pathlib.Path(plane))
                            plane_num += 1
                    if plane_num == 0:
                        raise(
                            Warning(f"No planes found in folder {subfolder}")
                            )
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
        ## Start setting up class
        # Index the folder(s) containing data from Sutie2p
        self.folder_path = pathlib.Path(folder_path) # Make pathlib object from folder path
        folder_content = utilities.file_handling.get_content(
            self.folder_path)
        self.plane_paths = index_suite2p_planes(self.folder_path)
        # Get name of folder and experiment
        self.folder_name = self.folder_path.name  # (?)
        # Index F file(s)
        self.f_index = index_from_path(self.plane_paths)
        self.fs = build_from_index(self.f_index)
        # Index .tiff file
        self.tiff_index = find_files(self.folder_path, "*_ch1", ".tiff")
        # Index and get trigger trace 
        self.trig_traces_index = find_files(self.folder_path, "*_ch2", ".npy")
        self.trigs = build_from_index((self.trig_traces_index))
        # Index and get iscell
        self.iscell_index = find_files(self.folder_path, "iscell", ".npy")
        self.iscells = build_from_index(self.iscell_index)
        # Index and get iscell
        self.stat_index = find_files(self.folder_path, "stat", ".npy")
        self.stats = build_from_index(self.stat_index)
        # Index and get ops
        self.ops_index = find_files(self.folder_path, "ops", ".npy")
        self.ops = build_from_index((self.ops_index))
        # Index 
        # Get experiment date
        
        ## Functions 
        # Corerct for temporal misalignment 
        def get_cell_positions(iscell):
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

            # Get numpy file's content 
            
            # Run through the numpy array, then the dictionary contained within, and list cell positions
            # return cell_positions
        
        def temporal_alignmnet(cell_positions, f_trace):
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
            # Upscale F trace temporarily to align

            # Downscale corrected F trace to original temporal precision
            
            # return corrected_f_traces
        self.panda = pd.DataFrame.from_dict(self.__dict__, orient='index')    
        print("Done")
    
# a = experiment(r"C:\Users\Simen\OneDrive\Universitet\PhD\test_tiffs_environment\mono_noUV_Rtect+20um")        
# a = experiment(r"D:\data_output\test_smh_environment_2")
a = experiment(r"C:\Users\Simen\OneDrive\Universitet\PhD\test_BC_out")

# """For testing on other computers"""

# Home
# ops = s2m.extract_data(r"D:\data_input\{}".format(inpt),
#                             r"D:\data_output",
#                             "{}".format(outpt),s
#                             512)

# Laptop
# s2m.extract_singleplane(r"C:\Users\Simen\Desktop\{}".format(inpt),
#                             r"C:\Users\Simen\Desktop",
#                             "{}".format(outpt),
#                             256)