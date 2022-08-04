# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 17:54:30 2021

@author: SimenLab
"""
import os
import pathlib
import numpy as np 
import processing_pypeline.readScanM as rsm

class get_stack:
    raise DeprecationWarning("Calling get_stack class from Import_Igor.py is depricated. Please import and call from pipeline_core.py instead.")
    curr_path = pathlib.Path.cwd()
    this_path = pathlib.Path(__file__).resolve().parent
    os.chdir(this_path)
    # This class is used in the .smh+.smp to .tiff conversion process, 
    # as well as for constructing the trigger signal. 
    def __init__(self, file_path):
        print("Getting stack...")
        self.file_path = pathlib.Path(file_path)
        self.data_directory = self.file_path.parent
        def change_dir(path_of_file):
            prev_path = pathlib.Path.cwd() #Make note of current (will be previous) working directory
            print(f"Making note of old path: {prev_path}")
            data_path = pathlib.Path(f'{path_of_file}')
            new_path = pathlib.Path(data_path).resolve().parent #Set directory to file location 
            print(f"Changing path to: {new_path}")
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
        print("Reverting path.")
        os.chdir(prev_path) # Go back where we came from
        print("Conversion complete.")
        
    def ch_arrays(self, res):
        ## Due to blanking artefact coming through ScanM, we have to crop the frame 
        ## down to the resolution we want. Because this artefact may be dependent
        ## on the selected resolution in ScanM, we will make a quick LUT to map 
        ## resolution down to crop. Fill in as needed...
        ## Lookup table, look up table
        resolution_map = { # Read tuple as left_crop, right_crop
            "128": (34, 32),
            "256": (28, 28),
            "512": (56, 56),
            "likely_fine": (35, 30),
            "no_crop": (),
            }
        # NOTE: Cropping note needed on y-axis
        if res == 128:
            self.channel1 = self.channel1[:, :, resolution_map["128"][0]:res+resolution_map["128"][1]]
        if res == 256:
            self.channel1 = self.channel1[:, :, resolution_map["256"][0]:res+resolution_map["256"][1]]
        if res == 512:
            self.channel1 = self.channel1[:, :, resolution_map["512"][0]:res+resolution_map["512"][1]]        # "Safe" setting, even if output res will be a bit lower than ideal
        if res == "likely_fine":
            self.channel1 = self.channel1[:, :, resolution_map["likely_fine"][0]:res+resolution_map["likely_fine"][1]]        # "Safe" setting, even if output res will be a bit lower than ideal
        print("Selected resolution:", res)
        return self.channel1, self.channel2

    def header_dict(stack_obj):
        header_dict = {
            'NumberOfFrames': stack_obj.frameN,
            'FrameCounter'  : stack_obj.frameC,
            'StimBufPerFr'  : stack_obj.frameB,
            'FrameHeight'   : stack_obj.frameH,
            'FrameWidth'    : stack_obj.frameW,
            }
        return header_dict
    def trigger_trace_frame(self):
        #Binarise the trigger using frame-wise percision (fast but impercise)
        ## Make an array of appropriate dimension
        trigger_trace_arr = np.zeros((1, self.channel2.shape[0]))[0]
        ## Loop through trigger image array
        for frame in range(self.channel2.shape[0]):
            if np.any(self.channel2[frame] > 1):
                trigger_trace_arr[frame] = 1
            #If trigger is in two consecutive frames, just use the first one so counting is correct
            if trigger_trace_arr[frame] == 1 and trigger_trace_arr[frame-1] == 1: 
                trigger_trace_arr[frame] = 0
        return trigger_trace_arr
    def trigger_trace_line(self):
        #Binarise the trigger using line-wise percision (slow but guaranteed percision)
        ## Make empty array that has dims frame_number x frame_size (for serialising each frame)
        trigger_trace_arr = np.empty((len(self.channel2), self.channel2[0].size))
        ## Loop through the input trigger array and serialise each frame
        for n, frame in enumerate(self.channel2):
            serial = frame.reshape(1, frame.size)
            ## Place that serialised data in its correct index
            trigger_trace_arr[n] = serial
        ## Our matrix is now an array of vectors containing serialised information from each frame
        ## Reshape this matrix into one long array (pixel x pixel-value)
        serial_trigger_trace = trigger_trace_arr.reshape(1, self.channel2.size)
        ## Then we binarise the serialised trigger data
        binarised_trigger_trace = np.where(serial_trigger_trace > 10000, 1, 0)[0]
        ## Boolean search for whether index n > n-1 (basically rising flank detection)
        trig_onset_serial = binarised_trigger_trace[:-1] > binarised_trigger_trace[1:]
        ## Get the frame indeces for trigger onset
        trig_onset_index = np.where(trig_onset_serial > 0)
        ## Then divide each number in trig_onset_index by the amount of lines
        trigg_arr_shape = self.channel2.shape
        lines_in_scan = trigg_arr_shape[1] * trigg_arr_shape[2]
        frame_of_trig = np.around(trig_onset_index[0]/lines_in_scan, 0)
        ## Convert back to frames 
        frame_number = len(self.channel2)
        trig_trace = np.zeros(frame_number)
        for i in frame_of_trig:
            trig_trace[int(i)] = 1
        return trig_trace



# Make trigger_trace --> This should be addedd to the get_stack class? 
def trigger_trace(trigger_arr):
    raise DeprecationWarning("Calling trigger trace from Import_Igor is depricated. Instead, please call either trigger_trace_frame or trigger_trace_line from Import_Igor.get_stack() object.. This may change in the future.")
    trigger_trace_arr = np.zeros((1, self.channel2.shape[0]))[0]
    for frame in range(self.channel2.shape[0]):
        if np.any(self.channel2[frame] > 1):
            trigger_trace_arr[frame] = 1
        #If trigger is in two consecutive frames, just use the first one so counting is correct
        if trigger_trace_arr[frame] == 1 and trigger_trace_arr[frame-1] == 1: 
            trigger_trace_arr[frame] = 0
        # else:
            # trigger_trace_arr[frame] = 0
    return trigger_trace_arr