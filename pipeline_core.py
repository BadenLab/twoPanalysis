# -*- coding: utf-8 -*-

import os
import pathlib
import time
import shutil
import warnings
import pathlib
import numpy as np
import tifffile
import suite2p
import pandas as pd
import pickle

import file_handling
import processing_pypeline.readScanM as rsm
import quantitative
import utilities
# import Import_Igor

class get_stack:
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
        #Binarise the trigger using line-wise percision (slower but guaranteed percision)
        ## Make empty array that has dims frame_number x frame_size (for serialising each frame)
        trigger_trace_arr = np.empty((len(self.channel2), self.channel2[0].size))
        ## Loop through the input trigger array and serialise each frame
        for n, frame in enumerate(self.channel2):
            ## Reshape frame into a series (vectorise it)
            serial = frame.reshape(1, frame.size)
            ## Place that serialised data in its correct index
            trigger_trace_arr[n] = serial
        ## Our matrix is now an array of vectors containing serialised information from each frame
        ## Reshape this matrix into one long array (1d, pixel-value per pixel)
        serial_trigger_trace = trigger_trace_arr.reshape(1, self.channel2.size)
        ## Interpolate down to line-precision 
        interpolated_trigger_trace = utilities.data.interpolate(serial_trigger_trace, self.channel2.shape[1] * self.channel2.shape[0])
        ## Then we binarise the serialised trigger data
        binarised_trigger_trace = np.where(interpolated_trigger_trace > 2500, 1, 0)[0]
        ## Boolean search for whether index n > n-1 (basically rising flank detection)
        trig_onset_serial = np.array(binarised_trigger_trace[:-1] > binarised_trigger_trace[1:])
        ## Interpolate down to line-precision 
        
        # ## Get the frame indeces for trigger onset
        # trig_onset_index = np.where(trig_onset_serial > 0)
        # ## Then divide each number in trig_onset_index by the amount of lines
        # trigg_arr_shape = self.channel2.shape
        # lines_in_scan = trigg_arr_shape[1] * trigg_arr_shape[2]
        # ## Round trig_onset_index/lines_in_scan to have 0 decimals
        # frame_of_trig = np.around(trig_onset_index[0]/lines_in_scan, 0)
        # ## Convert back to frames 
        # frame_number = len(self.channel2)
        # trig_trace = np.zeros(frame_number)
        # for i in frame_of_trig:
        #     trig_trace[int(i)] = 1
        return trig_onset_serial.astype(int)
## Algorithmically generate .tiffs and .npy (image and trigger) from Igor 
def gen_tiffs_from_igor(input_folder, output_folder, crop, **kwargs):
    input_folder = pathlib.Path(input_folder)
    output_folder = pathlib.Path(output_folder)
    img_count = 0
    ## Ensure no dataloss by skipping conversion where conversion has 
    ## already taken place 
    pre_existing_content = sorted(output_folder.rglob('*'))
    pre_existing_content_names = []
    for i in pre_existing_content:
        pre_existing_content_names.append(i.stem)
    for file in input_folder.glob('*.smp'):
        if file.with_suffix(".smh").exists() is False:
            raise FileNotFoundError(f"Could not find accompanying header (.smh) file for {file} in {input_folder}.")
        img_count += 1
        if file.stem in pre_existing_content_names:
            warnings.warn("Input files and output files have the same name, skipping conversion. Please manually delete files in output folder to force conversion.")
            continue
        else:
            file = pathlib.Path(file).resolve()
            ## Call get_stack class on image path to execute Igor conversion, and get access to class's methods
            img = get_stack(file)
            img_name = file.stem
            img_arr, trigger_arr = img.ch_arrays(crop)
            ## Select precision of trigger trace to algorithmically get the trigger trace out of trigger channel
            ### Default to line-wise precision
            if "trigger_precision" not in kwargs:
                ## If the kwarg doesn't exist, create it. (Convenient for labeling data in DataFrame)
                kwargs.update({'trigger_precision': "line"})
            ### Generate trace at the correct resolution and name trigger file accordingly
            if kwargs["trigger_precision"] == "line":
                trigger_trace = img.trigger_trace_line()
                trig_path = output_folder.joinpath(
                    img_name + "_line_res").with_suffix(".npy")
            if kwargs["trigger_precision"] == "frame":
                trigger_trace = img.trigger_trace_frame()
                trig_path = output_folder.joinpath(
                    img_name) + "_frame_res".with_suffix(".npy")


            print(kwargs)
            if "ignore_first_X_frames" not in kwargs:
                kwargs["ignore_first_X_frames"] = 0
            print(f"Skipping first {kwargs['ignore_first_X_frames']} frames")
            # if "ignore_last_X_frames" not in kwargs:
            #     kwargs["ignore_last_X_frames"] = len()
            tiff_path = output_folder.joinpath(
                img_name).with_suffix(".tiff")
            try:
                tifffile.imwrite(tiff_path, img_arr[kwargs["ignore_first_X_frames"]:])
            except FileNotFoundError:
                os.mkdir(tiff_path.parent)
            np.save(trig_path, trigger_trace[kwargs["ignore_first_X_frames"]*img_arr.shape[1]:])
            # if "pickle" in kwargs and kwargs["pickle"] is True:
            #     img_obj_path = output_folder.joinpath(
            #         "TEMP_pickle").with_suffix(".pickle")
            #     with open(img_obj_path, 'wb') as f:
            #         pickle.dump(img, f)
            del img, img_name, img_arr, trigger_arr, file
        if img_count == 0:
            raise TypeError("No Igor .smh or .smp files were identified!")

def prep_file_hierarchy(directory):
    tiff_paths = []
    trig_paths = []
    directory = pathlib.Path(directory)
    path_of_tiffs = sorted(directory.glob('*.tiff'))
    path_of_trigs = sorted(directory.glob('*.npy'))
    if len(path_of_trigs) == 0 and len(path_of_tiffs) == 0:
        print("No files found. Exiting.")
        return
    if len(path_of_trigs) == 0:
        print(path_of_trigs)
        warnings.warn("No .npy file detected. No trigger channel generated.")
        for tiff in path_of_tiffs:
            ### Step 3.1: Make folder with tiff filename
            new_single_plane_folder = directory.joinpath(
                tiff.stem)
            if new_single_plane_folder.exists() is False:
                new_single_plane_folder.mkdir()
            ### Step 3.2: Move tiff file into folder
            tiff_new_location = pathlib.Path(shutil.move(
                tiff, new_single_plane_folder))
            current_tiff_name = tiff_new_location.stem
            tiff_paths.append(current_tiff_name)
    else:
        for tiff, trig in zip(path_of_tiffs, path_of_trigs):
            ### Step 3.1: Make folder with tiff filename
            new_single_plane_folder = directory.joinpath(
                tiff.stem)
            if new_single_plane_folder.exists() is False:
                new_single_plane_folder.mkdir()
            ### Step 3.2: Move tiff file into folder
            tiff_new_location = pathlib.Path(shutil.move(
                tiff, new_single_plane_folder))
            current_tiff_name = tiff_new_location.stem
            tiff_paths.append(current_tiff_name)
            ### Step 3.3: Move .npy file (trigger trace) into folder 
            trig = pathlib.Path(trig)
            ### Step 3.2: Move trig file into folder
            trig_new_location = pathlib.Path(shutil.move(
                trig, new_single_plane_folder)).with_suffix(".npy")
            current_trig_name = trig_new_location.stem
            trig_paths.append(current_trig_name)
    path_of_tiffs = sorted(directory.rglob('*.tiff'))
    path_of_trigs = sorted(directory.rglob('*.npy'))
    return path_of_tiffs, path_of_trigs

def run_suite2p(ops, db):
    output_ops = suite2p.run_s2p(ops=ops, db=db)  # Run the actual algo...
    print("Initiating suite2p.run_s2p")
    # print(len(output_ops))
    output_ops_file = np.load(pathlib.Path(output_ops['save_path']).joinpath(
        'ops.npy'), allow_pickle=True).item()
    if output_ops_file.keys() != output_ops.keys():
        raise ValueError(
            "Keys in output_ops_file is different from keys in output_ops")
    return output_ops  # , output_ops_file

def extract_singleplane(input_folder, output_folder, crop, **kwargs):
    """
    Script for running Suite2p analysis on .tiffs with a single plane.
    E.g., every frame is from the same plane. The .tiffs are processed in
    sequence.

    Parameters
    ----------
    input_folder: Str or pathlib.Path object
        Folder from which Igor .smh's/.smp's are taken.
    save_dir (depricated): Str or pathlib.Path object
        Directory where outputs from Suite2p are stored.
    output_folder:
        Path where algorithm should output to.
    crop: Int
        Takes a single intiger and assumes it as squared (i.e. 256 (x 256), 512 (x 512), etc.)
    **path_of_ops: Path-like
        Path of options file to use. If not specified, uses inbuilt default.
    **path_of_classifier: Path-like
        Path of classifier file to use. If not specified, uses inbuilt default.
    Returns
    -------
    None.
    """
    # Define some handy inner functions
    ## Checks paths and returns True/False conditionally
    def probe_path(path, look_for):
        check_here = pathlib.Path(path)
        content = check_here.rglob(f'*/{look_for}')
        for i in content:
            if look_for in i.parts:
                target_content_present = True
                break
            else:
                target_content_present = False
            return target_content_present
    
    ## binaries, then save them in target folder.
    def copy_preexisting_tiffs():
        print("Identified .tiff files: Copying to output directory.")
        tiff_paths = list(pathlib.Path(input_folder).glob('*.tiff'))
        trig_paths = list(pathlib.Path(input_folder).glob('*.npy'))
        ### Copy over tiff files
        for input_file in tiff_paths:    
            shutil.copy2(input_file, output_folder.joinpath(input_file.name))
        ### Copy over npy files 
        for input_file in trig_paths:    
            shutil.copy2(input_file, output_folder.joinpath(input_file.name))
    ## Run Suite2p on each .tiff file in the file hieararchy 
    def tiff_f_extract(path_of_tiffs, **kwrags):
        tiff_num = len(path_of_tiffs)
        if tiff_num == 0:
            raise Warning("No .tiff files detected by tiff_f_extract().")
        else:
            print(f"Indexed {tiff_num} .tiff files. Running Suite2P API sequentially.")
        for tiff in path_of_tiffs:
            ### Point Suite2p to the right folder for analysis
            # needs to be a dictionary with a list of path(s)
            tiff_loc = pathlib.Path(tiff).parent
            db = {'data_path': [str(tiff_loc)], }
             # Select ops file
            if kwargs["path_of_ops"] is None or "path_of_ops" not in kwargs:
                ops = suite2p.default_ops()
                print("No ops file specified. Reverting to suite2p.default_ops()")
            if "path_of_ops" in kwargs and kwargs["path_of_ops"] is not None:
                loc_to_load = kwargs["path_of_ops"]
                ops = np.load(loc_to_load, allow_pickle=True)
                ops = ops.item()
            if "path_of_classifier" in kwargs:
                db["classifier_path"] = kwargs["path_of_classifier"]
            else:
                print("No classifier file specified. Reverting to in-built classifier (Suite2p default).")
            output_ops = run_suite2p(ops, db)
            # ops = suite2p.registration.metrics.get_pc_metrics(output_ops)
            # output_ops = run_suite2p(ops, db)
            # ops = suite2p.get_pc_metrics(ops)
    def select_data_extraction_type(input_folder):
        for file in sorted(input_folder.rglob('*')):
            suffix = file.suffix
            if suffix in [".tiff", ".tif"]:
                print(".tiff(s) identified. Copying them to ouput directory.")
                copy_preexisting_tiffs()
                break
            if file.suffix == ".smp" or file.suffix == ".smh":
                print("Igor file(s) identified. Initiating gen_tiffs_from_igor() function...")
                gen_tiffs_from_igor(input_folder, output_folder, crop, **kwargs)
                break
            else:
                raise FileNotFoundError("Appropriate filetype not found (tiff, Igor binary).")
    ## Check if folder already exists
    input_folder = pathlib.Path(input_folder)
    # save_dir = pathlib.Path(save_dir).resolve()
    output_folder = pathlib.Path(output_folder) 
    # final_destination = save_dir.joinpath(output_folder)
    print("Directory info") 
    print("- Save location:", output_folder)
    print("- Currently exists?", output_folder.exists())
    try:
        # Ideally...
        ## Simply make the directory:
        os.mkdir(output_folder.resolve())
        print(f"Target directory succesfully created: {output_folder}")
        print("Running data extraction algorithms.")
        ## Fill directory with data:
        select_data_extraction_type(input_folder)    
        ## Organise the file hieararchy
        tiff_paths, trig_paths = prep_file_hierarchy(output_folder)
        ## Run Suite2P on organised .tiff files
        tiff_f_extract(tiff_paths, 
                path_of_ops = kwargs["path_of_ops"],
                path_of_classifier = kwargs["path_of_classifier"]
                )
    except FileExistsError:
        print("Cannot create a directory when it already exists:", output_folder)
        ## Contingencies for handling pre-existing files
        if output_folder.exists() is True:
            any_check = any(output_folder.rglob('*'))
            suite2p_check = any(output_folder.rglob("suite2p"))
            print("Checking pre-existing content")
            print("- Content in directory?",  any_check)
            print("- Pre-existing Suite2p?", suite2p_check)
            ## If directory already exists but is empty, fill it with data
            if any_check is False and suite2p_check is False:
                print("Target directory is empty, running data extraction algorithms.")
                ## Fill directory with data:
                select_data_extraction_type(input_folder)
                ## Organise the file hieararchy
                tiff_paths, trig_paths = prep_file_hierarchy(output_folder)
            ## If Suite2P folders detected, abort to avoid overwriting previous analyses
            if suite2p_check is True:
                print(output_folder)
                warnings.warn("Suite2p-related content identified. Skipping this step.")
                return
            ## If .tiff files are present, index them
            elif any_check is True:
               tiff_paths = sorted(output_folder.rglob('*.tiff'))
               print(".tiff file(s) already exist here. Skipping conversion.")
               ## Organise the file hieararchy
               tiff_paths, trig_paths = prep_file_hierarchy(output_folder)
            ## Run Suite2P on organised .tiff files
            tiff_f_extract(tiff_paths, 
                path_of_ops = kwargs["path_of_ops"],
                path_of_classifier = kwargs["path_of_classifier"]
                )
        else:
            raise Warning("Unknown error when handling files.")

    print(f"Pipeline executed. Resulting files in {output_folder}")

class Experiment:
    def __init__(self, directory,**kwargs):
        # Function for identifying the folders representing each plane scanned
        def index_suite2p_planes(directory):
            """
            Parameters
            ----------
            directory : str or path
                The string or path-like (e.g. pathlib) that points to directory 
                for Suite2p

            Returns
            -------
            plane_index
                List of indeces for the planes folders
            """            
            
            plane_index = sorted(self.directory.rglob('**/plane*'))
            if not plane_index:
                raise ValueError("No Suite2p planes folders identified in directory.")
            return plane_index
        # Function for identifying relevant folders based on plane number
        def index_from_path(plane_path_list):
            """
            Args:
                plane_path_list (list of str or paths): _description_

            Returns:
            F_index (list): List of .npy files that contain F traces
            """
            path_index = []
            for plane in plane_path_list:
                new_path = plane.joinpath("F.npy")
                if new_path.exists() is True:
                    path_index.append(new_path)
                else:
                    raise(
                    Warning(f"No F.npy file at location {plane_path_list}")
                    )
            return path_index
        # Function for building the .npy files hierarchically based on file indexes
        def build_from_index(index_list):
            """build_from_index _summary_

            Parameters
            ----------
            index_list : _type_
                _description_

            Returns
            -------
            _type_
                _description_
            """
            fs_list = [np.load(f, allow_pickle=True) for f in index_list]
            return np.array(fs_list)
        # Quick function for finding a given file in all planes and sorting them
        def find_files(in_target, file_str, suffix_str):
            """find_files Simply globs files in target directory depending on file
            str and suffix str

            Parameters
            ----------
            in_target : str, path, or path-like
                The directory from which to glob
            file_str : str
                The name of the files to be globbed
            suffix_str : 
                The file extension aka suffix to look for (including '.')

            Returns
            -------
            List
                List of globbed files
            """
            return sorted(in_target.glob(f'**/{file_str}{suffix_str}'))
        
        def execute_data_construction_steps():
            # Index the folder(s) containing data from Sutie2p and then
            # make pathlib object from folder path
            self.directory = pathlib.Path(directory)
            self.folder_content = file_handling.get_content(
                self.directory)
            self.plane_paths = index_suite2p_planes(self.directory)
            # Make note of number of planes
            self.number_of_planes = len(self.plane_paths)
            # Create empty lists for building DataFrames later (list of dicts)
            list_of_dicts = []
            list_of_names = []
            # Return the immediate content of the directory (i.e. not recursively)
            directory_content = sorted(directory.glob('*'))
            # Check whether directory is the "final" dir, or a dir of sub-dirs (ignore .pickle
            # which may already be created...)
            check_dir = [i.is_file() for i in directory_content if i.suffix != ".pickle"]
            if any(check_dir):
                directory_content = [directory]
            # Loop through the expeirment folders and grab the relevant data
            print("Retrieving information from:")
            for folder in directory_content:
                ## Ignore files
                if folder.is_file() is True:
                    continue # Simply skips to next entry in directory_content
                ### NOTE: In the follow script, rglob (recursive glob) is used
                ### where files are stored in Suite2p sub-dirs several levels
                ### down. This ONLY works because 'folder' is specifically pointing
                ### to a directory containing a single Suite2p folder. Otherwise,
                ### this would cause absolute chaos (with tons of files indexed).
                print("  -  ", folder)
                folder_name = folder.name
                ## Index .tiff file
                tiff_index = sorted(folder.rglob('*.tiff'))
                ## Index and get F trace
                f_index = sorted(folder.rglob('*/plane*/F.npy'))
                fs = build_from_index(f_index)[0]
                ## Identify how many cells there are
                cell_number = fs.shape[0]
                ## Identify how many frames there are
                frame_number = fs.shape[1]
                ## Index and get trigger trace
                trig_index = sorted(folder.glob("*.npy"))
                trigs = build_from_index((trig_index))
                ## Index and get iscell
                iscell_index = sorted(folder.rglob('*/plane*/iscell.npy'))
                iscells = build_from_index(iscell_index)
                ## Index and get iscell
                stats_index = sorted(folder.rglob('*/plane*/stat.npy'))
                stats = build_from_index(stats_index)
                ## Index and get spks
                spks_index = sorted(folder.rglob('*/plane*/spks.npy'))
                spks = build_from_index(spks_index)
                ## Index and get ops
                ops_index = find_files(directory, "ops", ".npy")
                ops = build_from_index((ops_index))
                # Throw that data into a dictionary
                print("Generating info_dict")
                info_dict = {
                    "folder_name": folder_name,
                    # "plane_number": number_of_planes,
                    "f_index": f_index,
                    "fs": fs,
                    "cell_numbers" : cell_number,
                    "frame_number" : frame_number,
                    "tiff_index": tiff_index,
                    "trig_index": trig_index,
                    "trigs": trigs,
                    "iscell_index": iscell_index,
                    "iscells": iscells,
                    "stats_index": stats_index,
                    "stats": stats,
                    "spks_index": spks_index,
                    "spks": spks,
                    "ops_index": ops_index,
                    "ops": ops,
                    "notes" : str() # to be implemented (just pass as kwarg...?)
                }
                # Add that dictionary to the list
                list_of_dicts.append(info_dict)
                print("info_dict appended to list_of_dicts")
                # Also add the folder name to name_list for convenience
                list_of_names.append(folder_name)
            ## Make a DataFrame from the list of dictionaries (each row is one experiment)
            print(f"List of dictionaries with information from {self.directory}"
            "created and placed in DataFrame under self.panda.")
            self.panda = pd.DataFrame.from_dict(data=list_of_dicts).transpose()
            # Pass any kwargs information into dataframe for documentation.
            for i in kwargs:
                print(f"Passing {i} = {kwargs[i]} to kwargs")
                self.panda.loc[i] = kwargs[i]
            if 'x_res' not in kwargs or 'y_res' not in kwargs:
                raise warnings.warn("Kwarg 'x_res' and/or 'y_res missing. This determines several computations, but is not mandatory (computations will be skipped).")
            else:
                kwargs['x_res'] = int(kwargs['x_res'])
                kwargs['y_res'] = int(kwargs['y_res'])
            ## Generate statistics for trigger in given experiment 
            ### 1 Prepare empty dataframes
            self.panda.loc["trig_frames"] = 0
            self.panda.loc["num_trigs"] = 0
            self.panda.loc["trig_frames_interval"] = 0
            ### 2 Loop through and fill dataframes algorithmically
            for n, trig in enumerate(self.panda.loc["trigs"]):
                trig = trig[0]
                ## 3 Boolean search for whether index n > n-1
                trig_onset = trig[:-1] > trig[1:]
                ## 4 Insert trigger info into relevant indices
                self.panda.loc["trig_frames"][n] = np.where(trig_onset == 1)[0]
                self.panda.loc["num_trigs"][n] = len(self.panda.loc["trig_frames"][n])
                self.panda.loc["trig_frames_interval"][n] = np.average(np.gradient(self.panda.loc["trig_frames"][n]))
            #### 5 Optional: create timing info in seconds if kwarg for line duration is present 
            if 'line_duration' in kwargs:
                seconds_coeff = kwargs['y_res'] * kwargs['line_duration']
                self.panda.loc["trig_secs"] = self.panda.loc["trig_frames"] * seconds_coeff
                self.panda.loc["trig_intervals_secs"] = self.panda.loc["trig_frames_interval"] * seconds_coeff
            if 'line_duration' not in kwargs:
                raise warnings.warn("Kwarg 'line_duration' missing. Trigger timing information in seconds (['trig_secs'] and ['trig_secs_interval']) not generated.")
            ## Build averages algorithmically if specified to do so
            if "averages" in kwargs and kwargs["averages"] is True:
                print("Initiating algorithmic averaging")
                if "mode" not in kwargs:
                    raise ValueError("Missing **kwargs value 'mode', which determines"
                                    "averaging of f-traces. Please specify int value.")
                ## Initiate cells to overwrite information to (appends '0' to
                # all rows of experiment-columns)
                self.panda.loc["f_avgs"] = 0
                self.panda.loc["f_trials"] = 0
                self.panda.loc["trig_trials"] = 0
                self.panda.loc["trig_avgs"] = 0
                ## Get information from each experiment
                for i, (f, trg, md) in enumerate(
                                    zip(self.panda.loc["fs"], 
                                    self.panda.loc["trigs"], 
                                    self.panda.loc["mode"])):
                    ## Pass it to the averaging function
                    print("Averaging for experiment", self.panda.loc["folder_name"][i])
                    f_avg, f_trial, trig_trial, trig_avg = quantitative.average_signal(f, trg, md)
                    ## Insert the returned information into correct indeces
                    self.panda.loc["f_avgs"][i] = f_avg
                    self.panda.loc["f_trials"][i] = f_trial
                    self.panda.loc["trig_trials"][i] = trig_trial
                    self.panda.loc["trig_avgs"][i] = trig_avg
                self.avgs = self.panda.loc["f_avgs"]
                self.avg_trigs = self.panda.loc["trig_avgs"]
            ## Declare that DataFrame is complete
            print(f"DataFrame complete. Generating object-to-DataFrame shortcuts.")
            # Make each row accessible via object (i.e. "shortcuts")
            self.names = self.panda.loc["folder_name"]
            self.trigs = self.panda.loc["trigs"]
            self.fs = self.panda.loc["fs"] # 3 dims: Plane, cell, time
            self.frames = self.panda.loc["frame_number"]
            self.ops = self.panda.loc["ops"]
            self.stats = self.panda.loc["stats"]
            self.iscells = self.panda.loc["iscells"]
            self.spks = self.panda.loc["spks"]
            self.cell_numbers = self.panda.loc["cell_numbers"]
            self.notes = self.panda.loc["notes"]
            print("Shortcuts created (e.g. self.fs for self.panda.loc['fs']")
            # ## This just makes a dictionary, which may be pointless...
            # self.dict = self.__dict__

        # Generate the information for pandas DataFrame based on class input
        execute_data_construction_steps()
        print("Experiment object generated")
    
        """Functions to be...:"""
        # Corerct for temporal misalignment 
        def get_cell_positions(stats):
            """
            Run through the iscell.npy file for an experiment and index each cell position.

            Parameters
            ----------
            iscell : .npy 
                The iscell.npy file resulting from running Suite2p

            Returns
            -------
            cell_positions : numpy 2d-array
                Two-dimensional array containing (X, Y) and numerical cell-index  

            """
             
            stats[i][j][k]["ypix"]
            stats[i][j][k]["xpix"]
            # Run through the numpy array, then the dictionary contained within, and list cell positions
            # return cell_positions
        
           # return position_tuple 
        def f_temporal_alignmnet(cell_positions, f_trace):
            """
            Temporally align cell responses by 

            Parameters
            ----------
            cell_positions : TYPE
                DESCRIPTION.
            f_trace : TYPE
                DESCRIPTION.

            Returns
            -------
            corrected_f_traces : TYPE
                DESCRIPTION.

            """



#  def run_scripts(): ## This should move to main.py eventually. 
#         print(input_folder)
#         print(output_folder)
#         if skip_2p.value is False:
#             # This is the function that runs the data extraction pipeline:
#             pipeline_core.extract_singleplane(input_folder, 
#                                          output_folder, 
#                                          crop, 
#                                          path_of_ops = selected_ops_path, 
#                                          path_of_classifier = selected_classifier_path)
#         else:
#             print("Skipping Suite2p portion of pipeline.")
#             #If the user wishes to skip Suite2p portion of pipeline, they can simply call some lower-level functions for getting tiffs and creating a file hierarchy. 
#             pipeline_core.gen_tiffs_from_igor(input_folder, output_folder, crop)
#             pipeline_core.prep_file_hierarchy(output_folder)
#         if skip_obj.value is False:
#             print("Building Experiment-object.")
#             obj = pipeline_core.Experiment(
#                     output_folder, 
#                     averages = True, 
#                     mode = 30, 
#                     line_duration = line_duration_selection.value, 
#                     x_res = x_res_select.value, 
#                     y_res = y_res_select.value) #kwargs could easily be made into GUI items --> Work in progress... 
#             obj_save_location = output_folder.joinpath(f"{output_folder.name}.pickle")
#             print(f"Experiment object built, storing as {obj_save_location}")
#             with open(obj_save_location, 'wb') as f:
#                 pickle.dump(obj, f)