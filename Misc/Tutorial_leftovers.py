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