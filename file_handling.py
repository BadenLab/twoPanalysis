import pathlib

def load_experiment(f_path, trigger_path):
    """Loads up the files for F trace and trigger trace, taking the path for each respectively as input"    

    Args:
        f_path (str): Path in str of F trace numpy file 
        trigger_path (str): Path in str of trigger trace numpy file 

    Returns:
        f, trigger: The F and trigger as numpy arrays
    """        
    f = np.load(f_path, allow_pickle = True)
    trigger = np.load(trigger_path, allow_pickle = True)
    return f, trigger

# File-handling related
def get_Pathlib(path):
    pathlib_path = pathlib.Path(path)
    return pathlib_path

def get_content(folder_path):
    """
    Takes a folder path (in str or path-like) and returns the 

    Parameters
    ----------
    folder_path : TYPE
        DESCRIPTION.

    Returns
    -------
    folder_contents : TYPE
        DESCRIPTION.

    """
    folder = pathlib.Path(folder_path)
    folder_contents = list()
    for child in folder.iterdir():
        # print(child)
        folder_contents.append(child)
    # content = folder.iterdir()
    return folder_contents

def get_ops(path):
    ops =  np.load(path, allow_pickle=True)
    ops = ops.item()
    
def read_item(file_path):
    """
    Utility function for quickly and correctly importing complexnumpy items
    from Suite2p folders (iscell, stats, etc.).

    Parameters
    ----------
    file_path : TYPE
        DESCRIPTION.

    Returns
    -------
    ops : TYPE
        DESCRIPTION.

    """
    ops = np.load(file_path, allow_pickles=True)
    ops = ops.item()
    return ops