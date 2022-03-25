# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 14:36:21 2022

@author: SimenLab
"""

    # if "remove_border" in kwargs:
    #     if kwargs["remove_border"] == True:
    #         try:
    #             border = kwargs["border"]
    #             _left, _right, _top, _bottom = border[0], border[1], border[2], border[3]
    #         except:
    #             if "border" not in kwargs:
    #                 raise ValueError(
    #                     "Borders not defined. Border format:(Left, Right, Top, Bottom)")
    #             if type(border) is not list:
    #                 raise TypeError("kwarg 'border' must be list with four entries: [Left, Right, Top, Bottom] ")

    #     else:
    #         pass
    #         nFrames=(frameTotal-frameCounter)*frameBuffer
            
    #         c1=np.reshape(dataArray[0:nFrames*frameHeight*(frameWidth)],
    #                      (nFrames,frameHeight,frameWidth),
    #                      order="C")
    #         return c1
        
    # else:
    #     nFrames=(frameTotal-frameCounter)*frameBuffer
        
    #     c1=np.reshape(dataArray[0:nFrames*frameHeight*frameWidth],
    #                  (nFrames,frameHeight,frameWidth),
    #                  order="C")
    #     return c1