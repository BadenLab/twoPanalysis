# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 14:34:52 2022

Credits to: https://www.github.com/MouseLand/suite2p/blob/main/jupyter/run_suite2p_colab_2021.ipynb

@author: SimenLab
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import sys
import warnings
import tifffile
import suite2p
import matplotlib as mpl
import shutil
import time

import Import_Igor
import utilities


"""Currently hard-coded..."""
# import options
# import options_BC_testing

"""
Make plotting nice and consistent
"""
# mpl.rcParams.update({
#     'axes.spines.left': True,
#     'axes.spines.bottom': True,
#     'axes.spines.top': False,
#     'axes.spines.right': False,
#     'legend.frameon': False,
#     'figure.subplot.wspace': .01,
#     'figure.subplot.hspace': .01,
#     'figure.figsize': (18, 13),
#     'ytick.major.left': True,
# })
# cmap = mpl.cm.get_cmap("jet").copy()
# # jet = mpl.cm.get_cmap('jet')
# cmap.set_bad(color='k')

"""
TODO Need to crop the imaging file to the trigger channel already during the
Import_Igor phase... Maybe the class I wrote earler would be useful for this.
"""

"""
Set pipeline parameters
______________________________________________________________________________________________________________________________________________________________
TIP: Since it's common to change datasets and keep the same parameters for each
 dataset, some might find it useful to specify data-related arguments in db and
 pipeline parameters in op.

 See 'options.py'
"""
# ## Call default ops file
# ops = suite2p.default_ops()
# # ops['batch_size'] = 200 # we will decrease the batch_size in case low RAM on computer
# ops['threshold_scaling'] = 1 # we are increasing the threshold for finding ROIs to limit the number of non-cell ROIs found (sometimes useful in gcamp injections)
# ops['fs'] = 4 # sampling rate of recording, determines binning for cell detection
# ops['tau'] = 1.25 #was 1.25 # timescale of gcamp to use for deconvolution
# print(ops)
# # print(ops)
# # Initialise database (db) parameters

# db = {
#     'data_path': ['../data/test_data2'],
#     # 'save_path0': TemporaryDirectory().name,
#     # 'tiff_list': ['temp_tif.tiff'],
# }

## Legacy ops settings. Better to append/edit default ops than generating from scratch
# ops = {
#  'nplanes': 1,
#  'data_path': tif_save.with_suffix(".tiff"),
#  'save_path': r"C:\Users\SimenLab\OneDrive\Universitet\PhD\Python files\Git repos\2Panalysis\Data\Data dump",
#  'save_folder': r"C:\Users\SimenLab\OneDrive\Universitet\PhD\Python files\Git repos\2Panalysis\Data\ ",
#  'fast_disk': r"C:\Users\SimenLab\OneDrive\Universitet\PhD\Python files\Git repos\2Panalysis\Data\ ",
#  'nchannels': 0,
#  'keep_movie_raw': 1,
#  'look_one_level_down': 0,
#  }


"""
Run Suite2p on Data
______________________________________________________________________________________________________________________________________________________________
The suite2p.run_s2p function runs the pipeline and returns a list of output
 dictionaries containing the pipeline parameters used and extra data calculated
 along the way, one for each plane.
"""
# The ops dictionary contains all the keys that went into the analysis, plus
# new keys that contain additional metrics/outputs calculated during the
# pipeline run.


def gen_ops(ops, db):
    output_ops = suite2p.run_s2p(ops=ops, db=db)  # Run the actual algo...
    print("Initiating suite2p.run_s2p")
    # print(len(output_ops))
    output_ops_file = np.load(Path(output_ops['save_path']).joinpath(
        'ops.npy'), allow_pickle=True).item()
    if output_ops_file.keys() != output_ops.keys():
        raise ValueError(
            "Keys in output_ops_file is different from keys in output_ops")
    return output_ops  # , output_ops_file


def extract_singleplane(input_folder, save_dir, output_folder, crop):
    """
    Script for running Suite2p analysis on .tiffs with a single plane.
    E.g., every frame is from the same plane. The .tiffs are processed in
    sequence.

    Parameters
    ----------
    input_folder: Str or pathlib.Path object
        Folder from which Igor .smh's/.smp's are taken.
    save_dir: Str or pathlib.Path object
        Directory where outputs from Suite2p are stored.
    output_folder:
        Name of folder in save_dir algorithm should output to.
    crop: Int
        Takes a single intiger and assumes it as squared (i.e. 256 (x 256), 512 (x 512), etc.)
    ops_path: Path-like
        Path of options file to use.
    Returns
    -------
    None.

    """
    #Step 1: Make a folder under 'data_output' with Igor file's filename.without_suffix()
    ## Check if folder already exists
    input_folder = pathlib.Path(input_folder)
    save_dir = pathlib.Path(save_dir).resolve()
    output_folder = pathlib.Path(output_folder)
    final_destination = save_dir.joinpath(output_folder)
    other_stuff = False

    try:
        os.mkdir(final_destination.resolve())
    except FileExistsError:
        probe_path = pathlib.Path(save_dir).joinpath(output_folder)
        print("Data already exists in target output folder, checking if Suite2p-related")
        time.sleep(.25)
        # Check if folder already exists
        for child in probe_path.iterdir():
            if child == ((output_folder)):
                print("Suite2p analysis already exists here")
                sys.exit()
                # break
            if child == str(output_folder.joinpath("suite2p")):
                print("Here be pirates")
                sys.exit()
                other_stuff = True
                time.sleep(.1)
                # sys.exit()
                # break
    # Step 2: Get file, convert to TIF, place in folder
    ## Check if .tiff files have already been made
    if other_stuff is True:
        # Get names (stems, w/o suffix) of files in input folder
        check_list_input = []
        for file_in in input_folder.iterdir():
            check_list_input.append(pathlib.Path(file_in).stem)
        # Get names (stems) of files in output folder
        check_list_output = []
        for file_out in final_destination.iterdir():
            check_list_output.append(pathlib.Path(file_out).stem)
            "^ This returns everything in folder, so could just use this"
            "method to check if Suite2p already exists there... Saves time?"
        if any(file_stems in check_list_output for file_stems in check_list_input) is True:
            warnings.warn(
                "Input Igor binaries have same filenames as existing .tiffs in target output folder")
            time.sleep(1)
            tiff_count = len(sorted(final_destination.glob("*.tiff")))
            print(
                f"Found {tiff_count} pre-existing .tiffs. Running Suite2p on pre-existing .tiffs")
    elif other_stuff is False:
        _check_file_types = []
        tiff_count = 0
        for file in input_folder.iterdir():
            # print(file)
            _check_file_types.append(pathlib.Path(file).suffix)
            if ".tiff" in _check_file_types:
                warnings.warn(
                    ".tiff file(s) already exist here. Skipping .tiff conversion")
            if ".tif" in _check_file_types:
                warnings.warn(
                    ".tif file(s) already exist here. Skipping .tiff conversion")
            else:
                if file.suffix == ".smp":
                    tiff_count += 1
                    file = pathlib.Path(file).resolve()
                    img = Import_Igor.get_stack(file)
                    img_name = file.stem
                    img_arr, trigger_arr = Import_Igor.get_ch_arrays(img, crop)
                    trigger_trace = Import_Igor.trigger_trace(trigger_arr) # Algorithmically get the trigger trace out of trigger channel
                    # save_folder = pathlib.Path(r".\Data\data_output\{}".format(img_name)) # Bit more elegant than above
                    tiff_path = final_destination.joinpath(
                        img_name).with_suffix(".tiff")
                    trig_path = final_destination.joinpath(
                        img_name).with_suffix(".npy")
                    tifffile.imsave(tiff_path, img_arr)
                    np.save(trig_path, trigger_trace)
                    del img, img_name, img_arr, trigger_arr, file
                    """
                    TODO
                    Make shutil.move to a folder with the same file name (bascially what 
                    what happens in the function below) but do it earlier. This is so that
                    the tiff and the trig trace can be stored in the same folder as'suite2p'
                    
                    --> Currently doing this in tiff_f_extract... Move it here?
                    """
    # Step 3: For every .tiff in directory, run Suite2p on that file individually
    ## Get list of subdirs in dir
    subdir_paths = [f.path for f in os.scandir(
        final_destination) if f.is_dir()]
    ## Get list of tiff paths
    tiff_paths = list(final_destination.glob('*.tiff'))
    # if tiff_paths and subdir_paths have overlapping stems - suffix, ignore this analysis and prompt to check folders for Suite2p folder:

    ## For every tiff in tiff_paths, do the following:
    def tiff_f_extract(path_of_tiffs):
        for path_to_be_analysed in path_of_tiffs:
            path_to_be_analysed = pathlib.Path(path_to_be_analysed)
            ### Step 3.1: Make folder with tiff filename
            new_single_tiff_folder = final_destination.joinpath(
                path_to_be_analysed.stem)
            if new_single_tiff_folder.exists() == False:
                new_single_tiff_folder.mkdir()
            ### Step 3.2: Move tiff file into folder
            tiff_new_location = pathlib.Path(shutil.move(
                path_to_be_analysed, new_single_tiff_folder))
            ### Step 3.3: Point Suite2p to the right folder for analysis
            # needs to be a dictionary with a list of path(s)
            db = {'data_path': [str(new_single_tiff_folder)], }

            """Select ops file (this should not be hard-coded)..."""
            # ops = options.ops
            ops = options_BC_testing.ops
            # ops = np.load(ops_path, allow_pickle=True)
            # Step 4: Run Suite2p on this newly created folder with corresponding tiff file
            output_ops = gen_ops(ops, db)
            ops = suite2p.registration.metrics.get_pc_metrics(output_ops)
            output_ops = gen_ops(ops, db)
            # Alt step 5: Clean up files by moving them into the new folder position
            # then renaming them to their appropriate descriptions
            """This is a bit messy and could use a cleanup:"""
            current_tiff_name = tiff_new_location.stem
            current_tiff_final_location =  tiff_new_location.rename(
                tiff_new_location.with_stem(
                f"{current_tiff_name}_ch1").with_suffix(".tiff"))
            corresponding_trig_npy = path_to_be_analysed.with_suffix(".npy")
            trig_ch_name = path_to_be_analysed.stem
            # new_trig_path = corresponding_trig_npy.parent.joinpath(
            #     f"{trig_ch_name}_ch2")
            new_trig_path = corresponding_trig_npy.parent.joinpath(
                corresponding_trig_npy.stem, current_tiff_name
                ).with_stem(f"{trig_ch_name}_ch2").with_suffix(".npy")
            print(new_trig_path)
            shutil.move(corresponding_trig_npy, new_trig_path)
            # ops = suite2p.get_pc_metrics(ops)
    tiff_f_extract(tiff_paths)
# del output_ops, ops

    ## Data extraction
    # Step 5: Save info where it needs to go
    # f_cells, f_neuropils, spks = s2m.get_traces(output_ops)
    # stats_file, iscell, stats = s2m.detection(output_ops)
    # return output_ops



"Think a decorator is appropriate here, just need to work out how"
# @data_crop(f, trigger, start_buffer, end_buffer)


# fs, trig_trace = utilities.load_experiment(r"D:\data_output\test_tiffs_environment\mono_noUV_Rtect+20um\suite2p\plane0\F.npy", r"D:\data_output\test_tiffs_environment\mono_noUV_Rtect+20um.npy")

# def temporal_alignment(resolution, line_scan_speed, etc...):
#     """
#     TODO
#     Because lines are scanned sequentially, ROI responses will be temporally
#     misaligned (especially for higher resolutions). To re-enable the ability
#     to correlate the trigger channel with the imaging channel, signals from ROIs
#     will need to be temporally aligned.


#     Parameters
#     ----------


#     Returns
#     -------
#     int
#         DESCRIPTION.

#     """
#     return 1


# test1, test2, test3 = average_signal(fs, trig_trace, 30)


"""
Resulting files
______________________________________________________________________________________________________________________________________________________________
The output parameters can also be found in the "ops.npy" file. This is
especially useful when running the pipeline from the terminal or the graphical
interface. It contains the same data that is output from the python run_s2p()
function.
"""
# print(list(Path(output_op['save_path']).iterdir()))
# output_op_file = np.load(Path(output_op['save_path']).joinpath('ops.npy'), allow_pickle=True).item()
# output_op_file.keys() == output_op.keys()

"""
Vizualise resulting files
"""


def registration_viz(output_ops):
    plt.subplot(1, 4, 1)

    plt.imshow(output_ops['refImg'], cmap='gray', )
    plt.title("Reference Image for Registration")

    # maximum of recording over time
    plt.subplot(1, 4, 2)
    plt.imshow(output_ops['max_proj'], cmap='gray')
    plt.title("Registered Image, Max Projection")

    plt.subplot(1, 4, 3)
    plt.imshow(output_ops['meanImg'], cmap='gray')
    plt.title("Mean registered image")

    plt.subplot(1, 4, 4)
    plt.imshow(output_ops['meanImgE'], cmap='gray')
    plt.title("High-pass filtered Mean registered image")

# viz()


def offset_viz(output_ops):
    plt.figure(figsize=(18, 8))

    plt.subplot(4, 1, 1)
    plt.plot(output_ops['yoff'][:1000])
    plt.ylabel('rigid y-offsets')

    plt.subplot(4, 1, 2)
    plt.plot(output_ops['xoff'][:1000])
    plt.ylabel('rigid x-offsets')

    plt.subplot(4, 1, 3)
    plt.plot(output_ops['yoff1'][:1000])
    plt.ylabel('nonrigid y-offsets')

    plt.subplot(4, 1, 4)
    plt.plot(output_ops['xoff1'][:1000])
    plt.ylabel('nonrigid x-offsets')
    plt.xlabel('frames')
    plt.show()


"""
Detection
______________________________________________________________________________________________________________________________________________________________
"""
# ROIs are found by searching for sparse signals that are correlated spatially in
# the FOV. The ROIs are saved in stat.npy as a list of dictionaries which contain
# the pixels of the ROI and their weights (stat['ypix'], stat['xpix'], and stat['lam']).
# It also contains other spatial properties of the ROIs such as their aspect ratio
# and compactness, and properties of the signal such as the skewness of the
# fluorescence signal.
def detection(output_ops):
    stats_file = Path(output_ops['save_path']).joinpath('stat.npy')
    iscell = np.load(Path(output_ops['save_path']).joinpath(
        'iscell.npy'), allow_pickle=True)[:, 0].astype(int)
    stats = np.load(stats_file, allow_pickle=True)
    print(stats[0].keys())
    return stats_file, iscell, stats


def detection_viz(stats_file, iscell, stats, output_ops):
    n_cells = len(stats)

    h = np.random.rand(n_cells)
    hsvs = np.zeros(
        (2, output_ops["Ly"], output_ops["Lx"], 3), dtype=np.float32)

    for i, stat in enumerate(stats):
        ypix, xpix, lam = stat['ypix'], stat['xpix'], stat['lam']
        hsvs[iscell[i], ypix, xpix, 0] = h[i]
        hsvs[iscell[i], ypix, xpix, 1] = 1
        hsvs[iscell[i], ypix, xpix, 2] = lam / lam.max()

    from colorsys import hsv_to_rgb
    rgbs = np.array([hsv_to_rgb(*hsv)
                    for hsv in hsvs.reshape(-1, 3)]).reshape(hsvs.shape)

    plt.figure(figsize=(18, 18))
    plt.subplot(3, 1, 1)
    plt.imshow(output_ops['max_proj'], cmap='gray')
    plt.title("Registered Image, Max Projection")

    plt.subplot(3, 1, 2)
    plt.imshow(rgbs[1])
    plt.title("All Cell ROIs")

    plt.subplot(3, 1, 3)
    plt.imshow(rgbs[0])
    plt.title("All non-Cell ROIs")

    plt.tight_layout()


"""
Traces
______________________________________________________________________________________________________________________________________________________________
"""


def get_traces(output_ops):
    f_cells = np.load(Path(output_ops['save_path']).joinpath('F.npy'))
    f_neuropils = np.load(Path(output_ops['save_path']).joinpath('Fneu.npy'))
    spks = np.load(Path(output_ops['save_path']).joinpath('spks.npy'))
    f_cells.shape, f_neuropils.shape, spks.shape
    return f_cells, f_neuropils, spks
