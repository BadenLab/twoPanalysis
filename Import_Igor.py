# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 17:54:30 2021

@author: SimenLab
"""

"""

"""

###----- Needed during testing -----

#===== Benchmarking =====
# import timeit
# start = timeit.default_timer()

# import cProfile, pstats, io
# from pstats import SortKey
# pr = cProfile.Profile()
# pr.enable()
#===== Benchmarking =====


#==== OS module =====
import os
# abspath = os.path.abspath(__file__)
# dname = os.path.dirname(abspath)
# print(dname)
# os.chdir(dname)
# os.chdir("C:\\Users\\Simen\\OneDrive\\Universitet\\PhD\\Python files\\Modules\\twoPanalysis\\Example data")
#==== OS module =====

 
#==== Pathlib =======
import pathlib
curr_path = pathlib.Path.cwd()
this_path = pathlib.Path(__file__).resolve().parent
os.chdir(this_path)
# print(os.getcwd())
#==== Pathlib =======

###----- Needed during testing -----

import processing_pypeline.readScanM as rsm
# import numpy as np
import pathlib
import os
# import tifffile
import numpy as np 

class get_stack:
    # This class is used in the .smh+.smp to .tiff conversion process, 
    # as well as for constructing the trigger signal. 
    def __init__(self, file_path):
        print("Getting stack...")
        self.file_path = pathlib.Path(file_path)
        self.data_directory = self.file_path.parent
        def change_dir(path_of_file):
            prev_path = pathlib.Path.cwd() #Make note of current (will be previous) working directory
            print("Making note of old path: {}".format(prev_path))
            data_path = pathlib.Path(r'{}'.format(path_of_file)) #Construct data path
            new_path = pathlib.Path(data_path).resolve().parent #Set directory to file location 
            print("Changing path to: {}".format(new_path))
            os.chdir(new_path)#Change directory
            print("Done")
            return prev_path
        prev_path = change_dir(path_of_file = self.data_directory)
        print("Getting data from {}".format(self.file_path))
        self.filename = self.file_path.name
        print("Data retrieved.")
        ## This part of script provided by Andre Chagas 
        dicHeader = rsm.read_in_header(filePath = self.file_path.with_suffix('.smh'))#+".smh")
        print("Reading header...")
        self.frameN = int(dicHeader["NumberOfFrames"])
        self.frameC = int(dicHeader["FrameCounter"])
        self.frameB = int(dicHeader["StimBufPerFr"]) #Buffer between stimuli (mirror back into place)
        self.frameH = int(dicHeader["FrameHeight"])
        self.frameW = int(dicHeader["FrameWidth"])
        channel_dict = rsm.read_in_data(filePath=self.file_path.with_suffix('.smp'), header=dicHeader,
                                  readChan1=True, readChan2=True, 
                                  readChan3=True, readChan4=True)
        print("Header read.")
        print("Converting to serialised data...")
        #Convert data from serialized to frames
        self.channel1 = rsm.to_frame(channel_dict["chan1"], frameTotal=self.frameN, 
                      frameCounter=self.frameC, frameBuffer=self.frameB, 
                      frameHeight=self.frameH, frameWidth=self.frameW)

        self.channel2 = rsm.to_frame(channel_dict["chan2"], frameTotal=self.frameN, 
                      frameCounter=self.frameC, frameBuffer=self.frameB, 
                      frameHeight=self.frameH, frameWidth=self.frameW)
        # self.channels_interleved = np.empty((self.channel1.size + self.channel2.size,), dtype=self.channel1.dtype)
        print("Conversion complete.")
        """Use: https://numpy.org/doc/stable/reference/generated/numpy.dstack.html"""
        os.chdir(prev_path) # Go back where we came from
        print("Reverting path.")
        

def get_ch_arrays(stack_obj, res):
    # return exp_obj.channel1[:, 0:256, 35:256+35], exp_obj.channel2[:, 0:256, 35:256+35] 
    return stack_obj.channel1[:, 0:res, 35:res+35], stack_obj.channel2[:, 0:res, 35:res+35] 

def get_header_dict(stack_obj):
    header_dict = {
        'NumberOfFrames': stack_obj.frameN,
        'FrameCounter'  : stack_obj.frameC,
        'StimBufPerFr'  : stack_obj.frameB,
        'FrameHeight'   : stack_obj.frameH,
        'FrameWidth'    : stack_obj.frameW,
        }
    return header_dict
    
# Make trigger_trace --> This should be addedd to the get_stack class? 
def trigger_trace(trigger_arr):
    trigger_trace_arr = np.zeros((1, trigger_arr.shape[0]))[0]
    for frame in range(trigger_arr.shape[0]):
        if np.any(trigger_arr[frame] > 1):
            trigger_trace_arr[frame] = 1
        #If trigger is in two consecutive frames, just use the first one so counting is correct
        if trigger_trace_arr[frame] == 1 and trigger_trace_arr[frame-1] == 1: 
            trigger_trace_arr[frame] = 0
        # else:
            # trigger_trace_arr[frame] = 0
    return trigger_trace_arr

"""
______________________________Testing__________________________________________
"""

# test = make_stack(-5000, 10000, ((256, 256, 800)))
# input_file = r"C:\Users\SimenLab\OneDrive\Universitet\PhD\Python files\Git repos\2Panalysis\1_8dfp_ntc3_512_1.smp"
# a = get_stack(input_file)
# # img = get_stack(r"C:\Users\SimenLab\OneDrive\Universitet\PhD\Python files\Git repos\2Panalysis", "1_8dfp_ntc3_512_1")
# a = get_stack(input_file)

# img_for_tif = get_ch_arrays(img, 512)
# tifffile.imsave(r"C:\Users\SimenLab\OneDrive\Universitet\PhD\Python files\Suite2p tests\test.tiff", img_for_tif[0])

# img1 = get_stack("L2 expanding circles 3-700")
# img1 = get_stack("L5-20my l10x10 100um")

# def stack_average(stack):
#     for i in len(stack.shape[x]):
#         Get every pixel in X axis 
#         for i in len(stack.shape[y]):
#             Get every pixel in Y axis 
#             a = 1
#             return 
# img1.channel1[:, 1, 2] <-- The time axis


# img1 = get_stack("L6 grid whole screen 600")



#_____________________________
# stop = timeit.default_timer()
# print('Time (sec.): ', stop - start)
# print('Time (min.): ', (stop - start)/60)

#_____________________________
# pr.disable()
# s = io.StringIO()
# sortby = SortKey.CUMULATIVE
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.print_stats()
# print(s.getvalue())