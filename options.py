# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 14:57:52 2022

@author: SimenLab
"""

import suite2p
import numpy as np

## Call default ops file
ops = suite2p.default_ops()

"""The below is for testing purposes. Better practice is to save ops.npy some
where safe and seperate, then load is at the var ops each time...

Note that the vars not explicity changed below will be the default value.

Docs found here: https://www.suite2p.readthedocs.io/en/latest/settings.html#roi-detection 
"""

"""_________________________________MAIN SETTINGS___________________________"""
ops_name = "256x256 2fps GCaMP6 single tectum optimised"
ops = suite2p.default_ops()

# ops['threshold_scaling'] = 2 # we are increasing the threshold for finding ROIs to limit the number of non-cell ROIs found (sometimes useful in gcamp injections)
ops['fs'] = 2 # sampling rate of recording, determines binning for cell detection
ops['tau'] = 1.25 #was 1.25 # timescale of gcamp to use for deconvolution
# ops["frames_include"] = 128 #NOTE THIS NEEDS TO BE REMOVED LATER, TESTING ONLY
ops["classifier_path"] = r"C:\Users\SimenLab\OneDrive\Universitet\PhD\Python files\Git repos\2Panalysis\Classifiers\256x256 GCaMP6s single tectum.npy" # default: 0, otherwise pass path to classifier file

"""_________________________________OUTPUT SETTINGS_________________________"""
# if ops["nplanes"] > 1:
    # ops["combined"] = False
ops["nplanes"] = 1
ops["combined"] = False
ops["report_time"] = True

"""_________________________________REGISTRATION____________________________"""
# General 
ops["do_registration"] = True #default value, here to be explicit 
ops["nimg_int"] = 200 #(int, default: 200) how many frames to use to compute reference image for registration
ops["batch_size"] = 200 #(int, default: 200) how many frames to register simultaneously in each batch. This depends on memory constraints - it will be faster to run if the batch is larger, but it will require more RAM.
ops["smooth_sigma"] = 1 # (float, default: 1.15) standard deviation in pixels of the gaussian used to smooth the phase correlation between the reference image and the frame which is being registered. A value of >4 is recommended for one-photon recordings (with a 512x512 pixel FOV).
ops["smooth_sigma_time"] = 0 # (float, default: 0) standard deviation in time frames of the gaussian used to smooth the data before phase correlation is computed. Might need this to be set to 1 or 2 for low SNR data.
 
# Non-rigid registration
ops["nonrigid"] = True #(bool, default: True) whether or not to perform non-rigid registration, which splits the field of view into blocks and computes registration offsets in each block separately.
ops["block_size"] = [128, 128] #(two ints, default: [128,128]) size of blocks for non-rigid registration, in pixels. HIGHLY recommend keeping this a power of 2 and/or 3 (e.g. 128, 256, 384, etc) for efficient fft
ops["snr_thresh"] = 1.25 #(float, default: 1.2) how big the phase correlation peak has to be relative to the noise in the phase correlation map for the block shift to be accepted. In low SNR recordings like one-photon, I’d recommend a larger value like 1.5, so that block shifts are only accepted if there is significant SNR in the phase correlation.
ops["keep_movie_raw"] = True # (bool, default: False) whether or not to keep the binary file of the non-registered frames. You can view the registered and non-registered binaries together in the GUI in the “View registered binaries” view if you set this to True.
ops["maxregshift"] = .25
ops["th_badframes"] = 0.2
ops["keep_movie_raw"] = True
ops["two_step_registration"] = True # (int, default: 0) if 1,  run registration twice (for low SNR data). keep_movie_raw must be True for this to work.
ops["reg_tif"] = True

"""_________________________________ROI DETECTION___________________________"""
ops["roidetect"] = True #being explicit here
# ops["sparse_mode"] = False #Default is supposed to be False, but breaks if is not True (unless you change it, so defaults is probably not False...)
ops["spatial_scale"] = 0 # (int, default: 0), what the optimal scale of the recording is in pixels. if set to 0, then the algorithm determines it automatically (recommend this on the first try). If it seems off, set it yourself to the following values: 1 (=6 pixels), 2 (=12 pixels), 3 (=24 pixels), or 4 (=48 pixels).
ops["threshold_scaling"] = 2.5 #(int, default: 5) This is the most relevant SNR-parameter. Controls threshold for detecting ROIs (how much they stand out from noise). Higher = less ROIs
ops["max_overlap"] = 0.75 # (float, default: 0.75) ROIs with more than ops[‘max_overlap’] fraction of their pixels overlapping with other ROIs will be discarded. Therefore, to throw out NO ROIs, set this to 1.0.
ops["high_pass"] = 75 #(int, default: 100) running mean subtraction across time with window of size ‘high_pass’. Values of less than 10 are recommended for 1P data where there are often large full-field changes in brightness
ops["smooth_masks"] = True #smooth_masks: (bool, default: True) whether to smooth masks in final pass of cell detection. This is useful especially if you are in a high noise regime.
ops["max_iterations"] = 80 #The number of max it
ops["denoise"] = 1 # (int, default: 0) if 1 run PCA denoising on binned movie to improve ROI detection
# ops["max_iterations"]
# ops["nbinned"]

"""_________________________________SIGNAL EXTRACTION_______________________"""
ops["allow_overlap"] = False # whether or not to extract signals from pixels which belong to two ROIs. By default, any pixels which belong to two ROIs (overlapping pixels) are excluded from the computation of the ROI trace.
ops["connected"] = 0 
ops["min_neuropil_pixels"] = 20 # (int, default: 350) minimum number of pixels used to compute neuropil for each cell
ops["inner_neuropil_radius"] = 0 # inner_neuropil_radius: (int, default: 2) number of pixels to keep between ROI and neuropil donut

"""_________________________________SPIKE DECONVOLUTION_____________________"""
ops["spikedetect"] = 1 #(int, default: 1) simple on/off
ops["neucoeff"] = 0.1 # (float, default: 0.7) neuropil coefficient for all ROIs.
ops["baseline"] = 'maximin' #  (string, default ‘maximin’) how to compute the baseline of each trace. This baseline is then subtracted from each cell. 
# Baseline options:
# 'maximin' computes a moving baseline by filtering the data with a Gaussian of width ops[‘sig_baseline’] * ops[‘fs’], and then minimum filtering with a window of ops[‘win_baseline’] * ops[‘fs’], and then maximum filtering with the same window. 
# 'constant' computes a constant baseline by filtering with a Gaussian of width ops[‘sig_baseline’] * ops[‘fs’] and then taking the minimum value of this filtered trace. 
# 'constant_percentile' computes a constant baseline by taking the ops[‘prctile_baseline’] percentile of the trace.
ops["win_baseline"] = 60 # (float, default: 60.0) window for maximin filter in seconds
ops["sig_baseline"] = 15 #(float, default: 10.0) Gaussian filter width in seconds, used before maximin filtering or taking the minimum value of the trace, ops[‘baseline’] = ‘maximin’ or ‘constant’.
ops["prctile_baseline"] = 8 # (float, optional, default: 8) percentile of trace to use as baseline if ops[‘baseline’] = ‘constant_percentile’.

np.save(r"C:\Users\SimenLab\OneDrive\Universitet\PhD\Python files\Git repos\2Panalysis\Ops configurations\{}".format(ops_name), ops)

# ops["neuropil_extract"] = False #, BROKEN
# ops["preclassify"] = True # BROKEN apply classifier before signal extraction with proability threshold
# ops["sparse_mode"] = False, BROKEN