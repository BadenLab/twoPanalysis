# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 13:33:04 2022

@author: SimenLab
"""

import pathlib

def get_content(folder_path):
    folder = pathlib.Path(folder_path)
    folder_contents = list()
    for child in folder.iterdir(): 
        print(child)
        folder_contents.append(child)
    # content = folder.iterdir()
    return folder_contents
    
a = get_content(r"C:\Users\SimenLab\OneDrive\Universitet\PhD\Python files\Modules\twoPanalysis\utilities")