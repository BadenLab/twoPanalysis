# -*- coding: utf-8 -*-
"""
function to read igor files
"""

#import necessary libraries 
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

#import Andre's package from local file "processing_pypeline"
import processing_pypeline.readScanM as rsm
dir(rsm)
class read_igor():
    def collectData(file, triggermode):
        list= []
        #remove .smp/.smh
        filename = file.replace('.smh','').replace('.smp', '')
        if filename not in list:
            #Grab header info from smh file
            dicHeader = rsm.read_in_header(filePath = filename+'.smh')

            #Grab information from the header
            frameN = int(dicHeader["NumberOfFrames"])
            frameC = int(dicHeader["FrameCounter"])
            #Buffer between stimuli (mirror back into place)
            frameB = int(dicHeader["StimBufPerFr"])
            frameH = int(dicHeader["FrameHeight"])
            frameW = int(dicHeader["FrameWidth"])

            #Read in binary data
            #Output is dictionary where each key is one channel
            #Channel1 = green channel, channel 3? /2 = triggers
            output = rsm.read_in_data(filePath=filename+".smp", header=dicHeader,
                                  readChan1=True, readChan2=True, 
                                  readChan3=True, readChan4=True)

            #Convert data from serialized to frames
            #Only for channel 1
            frame1 = rsm.to_frame(output["chan1"], frameTotal=frameN, 
                              frameCounter=frameC, frameBuffer=frameB, 
                              frameHeight=frameH, frameWidth=frameW)

            #Convert data from channel 2 to detect triggers
            frame2 = rsm.to_frame(output["chan2"], frameTotal=frameN,
                                  frameCounter=frameC, frameBuffer=frameB,
                                  frameHeight=frameH, frameWidth=frameW)
            #which trigger mode to use??
            indexex, trigArray1 = rsm.trigger_detection(frameData=frame2,
                                                triggerLevel=220,
                                                triggerMode=triggermode)
            list.append(filename)
