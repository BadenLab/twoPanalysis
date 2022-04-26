# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:04:23 2022

@author: SimenLab

Taken from Thomas Euler's QDSpy package
"""

import numpy as np

def generateLinearLUT ():
  # Return a linear LUT (see setGammaLUT for details)
  #
  tempLUT = []
  for j in range(3):
    temp = list(range(256))
    temp = [float(v)/255.0 for v in temp]
    tempLUT.append(temp)

  newLUT = np.array(tempLUT)
  newLUT = (255*newLUT).astype(np.uint16)
  newLUT.byteswap(True)
  return newLUT

def generateInverseLUT ():
  # ... for testing purposes
  #
  tempLUT = []
  for j in range(3):
    temp = list(range(255,-1,-1))
    temp =[float(v)/255.0 for v in temp]
    tempLUT.append(temp)

  newLUT = np.array(tempLUT)
  newLUT = (255*newLUT).astype(np.uint16)
  newLUT.byteswap(True)
  return newLUT

a = generateLinearLUT()
b = generateInverseLUT()
b = b.T
np.savetxt(r"C:\Users\SimenLab\OneDrive - University of Sussex\Desktop\fff.txt", b, fmt = '%.0f', delimiter = ',')