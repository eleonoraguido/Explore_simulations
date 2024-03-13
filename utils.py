from dataclasses import dataclass
import uproot
import json
import numpy as np
import os
import math

@dataclass
class TreeData:
    """
    Represents the data from a single tree.

    Attributes:
    Ene_MC (np.ndarray): Energy from Monte Carlo simulation.
    Ene (np.ndarray): Reconstructed energy.
    Dist (np.ndarray): Distance.
    traces (np.ndarray): Traces vector.
    ID (np.ndarray): ID from standard deviation.
    t0 (np.ndarray): t0 value.
    theta (np.ndarray): Theta value.
    Stot (np.ndarray): Stot value.
    S1000 (np.ndarray): Shsize value.
    azimuth (np.ndarray): AzimuthSP value.
    Nstat (np.ndarray): Nstat value.
    """
    Ene_MC: np.ndarray
    Ene: np.ndarray
    Dist: np.ndarray
    traces: np.ndarray
    ID: np.ndarray
    t0: np.ndarray
    theta: np.ndarray
    Stot: np.ndarray
    S1000: np.ndarray
    azimuth: np.ndarray
    Nstat: np.ndarray


@dataclass
class ProcessedData:
    """
    Represents processed data from a single tree.

    Attributes:
    lgE_MC (np.ndarray): Processed energy from Monte Carlo simulation.
    lgE (np.ndarray): Processed reconstructed energy.
    Dist (np.ndarray): Normalized distance.
    traces (np.ndarray): Cumulative traces vector.
    ID (np.ndarray): ID from standard deviation.
    t0 (np.ndarray): t0 value.
    theta (np.ndarray): Theta value in degrees.
    lg_Stot (np.ndarray): Processed and rescaled Stot value.
    lg_S1000 (np.ndarray): Processed S1000 value.
    azimuth (np.ndarray): AzimuthSP value in degrees.
    Nstat (np.ndarray): Normalized Nstat value.
    label (int): label (0 or 1)
    """
    lgE_MC: np.ndarray
    lgE: np.ndarray
    Dist: np.ndarray
    traces_cum: np.ndarray
    ID: np.ndarray
    t0: np.ndarray
    theta: np.ndarray
    lg_Stot: np.ndarray
    lg_S1000: np.ndarray
    azimuth: np.ndarray
    Nstat: np.ndarray
    label: int



def rescale_Stot(x_list, Snorm):
    """
    Rescale Stot values in a list.

    Parameters:
    x_list (list): A list of Stot values.
    Snorm (float): Normalization factor.

    Returns:
    list: A list of rescaled Stot values.
    """
    return [np.log10(xi + 1.0) / math.log10(Snorm + 1) for xi in x_list]




def process_data(tree_data_input, sel_theta = False, label_val=1, Snorm=100):
    """
    Process data from a single tree.

    Parameters:
    tree_data (TreeData): An instance of TreeData containing data arrays from a single tree.
    sel_theta (bool, optional): if true a selection of vertical events is performed (theta < 60°)
    label_val (int, optional): It distinguish between signal (1) and background (0) data
    Snorm (float, optional): Normalization factor for Stot (default is 100).

    Returns:
    ProcessedData: An instance of ProcessedData containing processed data arrays.
    """

    if(sel_theta):
        # Filter data based on theta
        filtered_indices = np.where(tree_data_input.theta < np.radians(60))[0]
        tree_data = TreeData(
            Ene_MC=tree_data_input.Ene_MC[filtered_indices],
            Ene=tree_data_input.Ene[filtered_indices],
            Dist=tree_data_input.Dist[filtered_indices],
            traces=tree_data_input.traces[filtered_indices],
            ID=tree_data_input.ID[filtered_indices],
            t0=tree_data_input.t0[filtered_indices],
            theta=tree_data_input.theta[filtered_indices],
            Stot=tree_data_input.Stot[filtered_indices],
            S1000=tree_data_input.S1000[filtered_indices],
            azimuth=tree_data_input.azimuth[filtered_indices],
            Nstat=tree_data_input.Nstat[filtered_indices]
        )
    else:
        tree_data = tree_data_input

    
    # Process Nstat
    Nstat = np.log10(tree_data.Nstat)

    lg_Stot = np.array([rescale_Stot(tree_data.Stot, Snorm)])
    lg_Stot = np.squeeze(lg_Stot)
  
    
    # Convert theta to degrees
    theta_deg = np.degrees(tree_data.theta)
    
    # Process S1000
    lg_S1000 = np.log10(tree_data.S1000)
    
    # Process energy (Ene)
    lgE = np.log10(tree_data.Ene)
    
    # Process energy from Monte Carlo simulation (Ene_MC)
    lgE_MC = np.log10(tree_data.Ene_MC)
    
    # Convert azimuth to degrees
    azimuth_deg = np.degrees(tree_data.azimuth)
    
    # Normalize Dist
    Dist_1500 = tree_data.Dist / 1500
    mean = np.mean(Dist_1500)
    Dist_norm = Dist_1500 - mean
 
    
    # Create a copy of the traces array to avoid modifying the original data
    traces = np.copy(tree_data.traces)
    
    # Normalize traces
    traces_cum = traces[:, :, :150]
    for j, event in enumerate(traces_cum):
        for i, stat in enumerate(event):
            stat = np.cumsum(stat)
            traces_cum[j][i] = stat / np.max(stat)


    ProcessedTree = ProcessedData(
        lgE_MC=lgE_MC,
        lgE=lgE,
        Dist=Dist_norm,
        traces_cum=traces_cum,
        ID=tree_data.ID,
        t0=tree_data.t0,
        theta=theta_deg,
        lg_Stot=lg_Stot,
        lg_S1000=lg_S1000,
        azimuth=azimuth_deg,
        Nstat=Nstat,
        label=label_val
    )

    return ProcessedTree


def read_tree(file_path):
    """
    Read data from a ROOT file containing a single tree.

    Parameters:
    file_path (str): The path to the ROOT file.

    Returns:
    TreeData: instance of TreeData containing arrays of data extracted from the tree
    """
    # Open the ROOT file
    file = uproot.open(file_path)

    # Access the tree (assuming there's only one tree)
    tree = file[file.keys()[0]]

    # Read branches
    branches = tree.arrays()

    # Create a TreeData object to hold the data
    tree_data = TreeData(
        Ene_MC=np.array(branches["E_MC"].tolist()),
        Ene=np.array(branches["E_SD"].tolist()),
        Dist=np.array(branches["Dist"].tolist()),
        traces=np.array(branches["traces_vec"].tolist()),
        ID=np.array(branches["ID_SD"].tolist()),
        t0=np.array(branches["t0"].tolist()),
        theta=np.array(branches["theta"].tolist()),
        Stot=np.array(branches["Stot"].tolist()),
        S1000=np.array(branches["Shsize"].tolist()),
        azimuth=np.array(branches["azimuthSP"].tolist()),
        Nstat=np.array(branches["Nstat"].tolist())
    )

    return tree_data



def read_config_plot(config_file):
    """
    Read input and output file paths from a JSON configuration file.

    Parameters:
    config_file (str): The path to the JSON configuration file.

    Returns:
    tuple: A tuple containing the input and output file paths:
        - input_file (str): The path to the input ROOT file.
        - output_file (str): The path to the output directory.
        - theta_cut (bool): if true only events with theta < 60° are selected.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config.get('input_file'), config.get('output_file'), config.get('theta_cut')


def read_config_dataset(config_file):
    """
    Read input and output file paths from a JSON configuration file.

    Parameters:
    config_file (str): The path to the JSON configuration file.

    Returns:
    tuple: A tuple containing the input and output file paths:
        - input_file1 (str): The path to the first input ROOT file.
        - input_file2 (str): The path to the second input ROOT file.
        - output_file (str): The path to the output directory.
	    - dataset_name (str): The name of the data set file that will be created.
        - theta_cut (bool): if true only events with theta < 60° are selected.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config.get('input_file1'), config.get('input_file2'), config.get('output_file'), config.get('dataset_name'), config.get('theta_cut')


def read_config_onepart_dataset(config_file):
    """
    Read input and output file path from a JSON configuration file.

    Parameters:
    config_file (str): The path to the JSON configuration file.

    Returns:
    tuple: A tuple containing the input and output file paths:
        - input_file (str): The path to the input ROOT file.
        - output_file (str): The path to the output directory.
	    - dataset_name (str): The name of the data set file that will be created.
        - theta_cut (bool): If true only events with theta < 60° are selected.
        - label (int): Label assigned to the events in the file (0 background, 1 signal).
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config.get('input_file'), config.get('output_file'), config.get('dataset_name'), config.get('theta_cut'), config.get('label')

    



def read_config_type(config_file):
    """
    Read the type of config file from the JSON configuration file.

    Parameters:
    config_file (str): The path to the JSON configuration file.

    Returns:
    config_type (str): The type of configuration file.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config.get('config_type')






def extract_particle_name(input_file):
    """
    Extracts the particle name from the input file path.

    Parameters:
    input_file (str): The input file path.

    Returns:
    str: The particle name extracted from the input file path.
    """
    # Split the input file path by "/"
    parts = input_file.split("/")
    
    # Get the last part of the path
    last_part = parts[-1]
    
    # Split the last part by "_"
    sub_parts = last_part.split("_")
    
    # Get the first part after the split
    particle_name = sub_parts[0]
    
    return particle_name
