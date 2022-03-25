# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 17:52:00 2021

@author: SimenLab
"""

import pandas as pd

def Data_getter(file_location):
    """A function which gets and prepares data from CSV files, as well as 
    returning some additional params like an number of ROIs and their 
    corresponding labels.

    Parameters
    ---------
    File: 
        The directory of a CSV .txt file containing the data, with each ROI
        represented as individual columns in the file.
    
    Returns
    -------
    Stimulus DataFrame, stimulus array (as an alternative), number of
    ROIs, and their labels.
    """
    file = file_location
    avgs_data = pd.read_csv((file), sep="\t", header=None, engine="python")

    ##Label data
    ROI_num = avgs_data.shape[1]

    # Consider making data loading its own function?
    "Can put optional manipulations to data here"

    # pd.read_csv((file), sep="\t", header=None) #, names = labels)
    averages_dataframe = avgs_data
    avgerages_array = pd.DataFrame.to_numpy(averages_dataframe)

    return averages_dataframe, avgerages_array, ROI_num

class Data:        
    def Data_getter(self):
            """A function which gets and prepares data from CSV files, as well as 
            returning some additional params like an number of ROIs and their 
            corresponding labels.
        
            Parameters
            ---------
            File: 
                The directory of a CSV .txt file containing the data, with each ROI
                represented as individual columns in the file.
            
            Returns
            -------
            Stimulus DataFrame, stimulus array (as an alternative), number of
            ROIs, and their labels.
            """
            file = self.file_location
            avgs_data = pd.read_csv((file), sep="\t", header=None, engine="python")

            ##Label data
            ROI_num = avgs_data.shape[1]

            # Consider making data loading its own function?
            "Can put optional manipulations to data here"

            # pd.read_csv((file), sep="\t", header=None) #, names = labels)
            averages_dataframe = avgs_data
            avgerages_array = pd.DataFrame.to_numpy(averages_dataframe)

            return averages_dataframe, avgerages_array, ROI_num
        
            Retrieved_data = Data_getter(self.storage_info)
            self.data = Retrieved_data[0]
            self.ROI_num = Retrieved_data[2]