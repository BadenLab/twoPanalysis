# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:13:31 2022

@author: SimenLab
"""
#Local imports:
import Suite2me as s2m
import utilities

# Environment imports:
import pathlib
import numpy as np 

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
                                           # r"C:\Users\Simen\OneDrive\Universitet\PhD\test_tiffs_environment\mono_noUV_Rtect+20um.npy")


fs, trig_trace = utilities.file_handling.load_experiment(r"D:/data_output/test_smh_environment_2/mono_noUV_Rtect+20um/suite2p/plane0/F.npy", 
                                            r"D:\data_output\test_smh_environment_2\mono_noUV_Rtect+20um.npy")

inpt = "test_BCs"
outpt = "test_BCs"

# # Work
# ops = s2m.extract_singleplane(r"D:\data_input\{}".format(inpt),
#                             r"D:\data_output",
#                             "{}".format(outpt), 
#                             256)

"""Work in progress:"""
# the idea with this is to load an experiment object which contains all the
# relevant information, including the extracted ROIs 
class experiment:
    def __init__(self, folder_path):
        ## Setup 
        # Make pathlib object from folder path
        self.folder_path      = pathlib.Path(folder_path)
        self.folder_content   = utilities.file_handling.get_content(self.folder_path)
        # Get name of folder and experiment  
        self.folder_name      = self.folder_path.name # (?)

        # Index the folder containing data from Sutie2p
        for subfolder in self.folder_path.iterdir():
            
            "utilities files get content"
            
            # subfolder = pathlib.Path(subfolder)
            print(subfolder)
            # for child in subfolder.iterdir():
                # print(child)
            
            # if subfolder.is_dir() == True:
            #     if list(subfolder.glob('*')) == 'suite2p':
            #         self.s2p_path = self.folder_path.joinpath(subfolder)
                # self.experiment_name  = self.s2p_path.name 
        # Index F file 
        # self.f_traces_path = self.s2p_path.joinpath("{}.npy".format(self.experiment_name))
        
        # Index and get trigger trace 
        # self.trig_traces_path = self.folder_path.parent.joinpath("{}.npy".format(self.experiment_name))
        # self.trig_trace       = np.load(self.trig_traces_path)
        
        
        
        # Index iscell
        # self.iscell = pathlib.Path()
        
        # Index ops
        
        # Index stat

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
        # def 
        
        # 
    
# a = experiment(r"C:\Users\Simen\OneDrive\Universitet\PhD\test_tiffs_environment\mono_noUV_Rtect+20um")        
a = experiment(r"D:\data_output\test_smh_environment_2")        


"""For testing on other computers"""
#Home
# ops = s2m.extract_data(r"D:\data_input\{}".format(inpt),
#                             r"D:\data_output",
#                             "{}".format(outpt),s
#                             512)

#Laptop
# s2m.extract_singleplane(r"C:\Users\Simen\Desktop\{}".format(inpt),
#                             r"C:\Users\Simen\Desktop",
#                             "{}".format(outpt),
#                             256)
#                             # r"C:\Users\Simen\Desktop\BC testing.npy")