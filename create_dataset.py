import numpy as np
from typing import List
from utils import ProcessedData

def create_info_event(processed_data: ProcessedData) -> np.ndarray:
    """
    Create info_event for a single ProcessedData object.

    Parameters:
    -----------
    processed_data : ProcessedData
        An instance of ProcessedData containing processed data arrays.

    Returns:
    -----------
    np.ndarray
        An array containing concatenated information from processed_data attributes
        (theta, lg_S1000, Nstat).
    """
    return np.column_stack((processed_data.theta, processed_data.lg_S1000, processed_data.Nstat))



def shuffle_arrays(arrays: List[np.ndarray], set_seed: int = 55) -> None:
    """
    Shuffles arrays in-place, in the same order, along axis=0

    Parameters:
    -----------
    arrays : List of NumPy arrays.
        List containing arrays to be shuffled.
    set_seed : int, optional
        Seed value if int >= 0, else seed is random. Default is 55.
    """
    assert all(len(arr) == len(arrays[0]) for arr in arrays)
    seed = np.random.randint(0, 2**(32 - 1) - 1) if set_seed < 0 else set_seed

    for arr in arrays: 
        rstate = np.random.RandomState(seed)
        rstate.shuffle(arr)

def merge_and_shuffle(output_dir: str, dataset_name: str, processed_data_list: List[ProcessedData], set_seed: int = 55) -> None:
    """
    Merge ProcessedData objects and shuffle them while preserving the correspondence between arrays.

    Parameters:
    -----------
    output_dir : str
        Directory path where the merged and shuffled dataset will be saved.
    dataset_name : str
        Name of the dataset file.
    processed_data_list : List[ProcessedData]
        List of ProcessedData objects to be merged and shuffled.
    set_seed : int, optional
        Seed value for shuffling. Default is 55.

    Returns:
    -----------
    None
    """
    
    #processed_data_list = [data for data in processed_data_list if data.label == 1] #filter only data with label 0

    # Create info_event for each ProcessedData object
    info_events = np.concatenate([create_info_event(data) for data in processed_data_list])

    # Extract relevant arrays and concatenate them along the existing axis
    traces = np.concatenate([data.traces_cum for data in processed_data_list])
    labels = np.concatenate([np.full_like(data.lgE_MC, data.label) for data in processed_data_list])
    Dist_norm = np.concatenate([data.Dist for data in processed_data_list])
    azimuth = np.concatenate([data.azimuth for data in processed_data_list])
    lg_Stot = np.concatenate([data.lg_Stot for data in processed_data_list])

    Dist_norm = Dist_norm.reshape(Dist_norm.shape[0], Dist_norm.shape[1], 1)
    azimuth = azimuth.reshape(azimuth.shape[0], azimuth.shape[1], 1)
    lg_Stot = lg_Stot.reshape(lg_Stot.shape[0], lg_Stot.shape[1], 1)
    traces = traces.reshape(traces.shape[0], traces.shape[1], traces.shape[2], 1)
    info_events = info_events.reshape(info_events.shape[0],info_events.shape[1], 1)
    labels = labels.reshape(labels.shape[0],)
   
    # Shuffle arrays while preserving correspondence
    arrays = shuffle_arrays([traces, labels, Dist_norm, info_events, azimuth, lg_Stot], set_seed=set_seed)
    
    np.savez_compressed(output_dir+dataset_name+'.npz', traces=traces, dist=Dist_norm, Stot=lg_Stot, azimuthSP=azimuth, info_event=info_events, labels=labels)
    print("File has been created as "+dataset_name+'.npz in '+output_dir)



def load_npz_file(output_dir: str, dataset_name: str)-> None:
    """
    Load the compressed NumPy file and check its content.

    Parameters:
    -----------
    output_dir : str
        Directory path where the dataset file is located.
    dataset_name : str
        Name of the dataset file.

    Returns:
    -----------
    None
    """
    # Load the compressed NumPy file
    data = np.load(output_dir + dataset_name+'.npz')

    print("Try to load it and check the content...")
    # Check the keys of the loaded data
    print(" Keys in the compressed file:", data.files)

    # Access individual arrays by their keys
    traces = data['traces']
    dist = data['dist']
    Stot = data['Stot']
    azimuthSP = data['azimuthSP']
    info_event = data['info_event']
    labels = data['labels']

    print(" Shapes of the arrays:")
    print(" Traces:", traces.shape)
    print(" Dist:", dist.shape)
    print(" Stot:", Stot.shape)
    print(" AzimuthSP:", azimuthSP.shape)
    print(" Info_event:", info_event.shape)
    print(" Labels:", labels.shape)

    data.close()