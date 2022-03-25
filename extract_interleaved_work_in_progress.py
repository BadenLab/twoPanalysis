# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 15:20:25 2022

@author: SimenLab
"""

def extract_interleaved(input_folder, save_dir, output_folder, crop, **kwargs):    # output_folder
    """
    Script for running Suite2p analysis on .tiffs with interleaved planes 
    E.g., every n-th frame is from the n-th plane.
    
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
    Returns
    -------
    None.

    """
    if "res" in kwargs:
        resolution = kwargs["res"]
    else:
        resolution = 256
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
        # Check if folder already exists
        for child in probe_path.iterdir():
            if child == ((output_folder)):
                print("Suite2p analysis already exists here")
                sys.exit()
                # break
            else:
                warnings.warn(
                    "Other content exists in {}".format(
                        pathlib.Path(save_dir).joinpath(output_folder)
                    )
                )
                other_stuff = True
                # sys.exit()
                # break
    # Step 2: Get file, convert to TIF, place in folder
    """TODO Need to re-write to interleave frames"""
    
    ## Check if .tiff files have already been made
    if other_stuff == True:
        print("Target folder contains .tiffs, checking if identical to import files")
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
        if any(file_stems in check_list_output for file_stems in check_list_input) == True:
            warnings.warn("Input Igor binaries have same filenames as existing .tiffs in target output folder")
            tiff_count = len(sorted(final_destination.glob("*.tiff")))
            print("Found {} pre-existing .tiffs. Running Suite2p on pre-existing .tiffs".format(tiff_count))
            # img = 
            # sys.exit()
        # raise Warning(
        #     "Stuff is already in target directory. Delete that and try again.")
        # sys.exit()
    else:
       _check_file_types = []
       tiff_count = 0
       for file in input_folder.iterdir():
           # print(file)
           _check_file_types.append(pathlib.Path(file).suffix)
    #     "Here need a check if .tiff files already exist to "
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
                   img_arr, trigger_arr = Import_Igor.get_ch_arrays(img, resolution)
                   # save_folder = pathlib.Path(r".\Data\data_output\{}".format(img_name)) # Bit more elegant than above
                   tifffile.imsave(final_destination.joinpath(
                       img_name).with_suffix(".tiff"), img_arr)    
                   del img, img_name, img_arr, trigger_arr, file
    
    ops = options.ops
    if tiff_count > 1:
        ops["nplanes"] = tiff_count
    ops["classifier_path"] = 0
    # Step 3: Point Suite2p to the right folder for analysis
    # needs to be a dictionary with a list
    db = {'data_path': [str(final_destination)], }
    # Step 4: Run Suite2p
    output_ops = gen_ops(ops, db)
    ops = suite2p.registration.metrics.get_pc_metrics(output_ops)
    output_ops = gen_ops(ops, db)
    # ops = suite2p.get_pc_metrics(ops)
    # del output_ops, ops
    
    ## Data extraction
    # Step 5: Save info where it needs to go
    # f_cells, f_neuropils, spks = s2m.get_traces(output_ops)
    # stats_file, iscell, stats = s2m.detection(output_ops)
    return output_ops
    # return f_cells, f_neuropils, spks, stats_file, iscell, stats, ops, db, output_ops, img_arr, trigger_arr, header_info
#fcells, isfile, etc... Suite2p related stuff. The rest can be done in-function