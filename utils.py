from dataclasses import dataclass
import uproot
import json
import numpy as np
import os

@dataclass
class TreeData:
    """
    Represents the data from a single tree.

    Attributes:
    Ene_MC (np.ndarray): Energy from Monte Carlo simulation.
    Ene (np.ndarray): Energy from standard deviation.
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
    lgE (np.ndarray): Processed energy from standard deviation.
    Dist (np.ndarray): Normalized distance.
    traces (np.ndarray): Normalized traces vector.
    ID (np.ndarray): ID from standard deviation.
    t0 (np.ndarray): t0 value.
    theta (np.ndarray): Theta value in degrees.
    lg_Stot (np.ndarray): Processed and rescaled Stot value.
    lg_S1000 (np.ndarray): Processed S1000 value.
    azimuth (np.ndarray): AzimuthSP value in degrees.
    Nstat (np.ndarray): Processed Nstat value.
    """
    lgE_MC: np.ndarray
    lgE: np.ndarray
    Dist: np.ndarray
    traces: np.ndarray
    ID: np.ndarray
    t0: np.ndarray
    theta: np.ndarray
    lg_Stot: np.ndarray
    lg_S1000: np.ndarray
    azimuth: np.ndarray
    Nstat: np.ndarray



def process_data(tree_data, Snorm=100):
    """
    Process data from a single tree.

    Parameters:
    tree_data (TreeData): An instance of TreeData containing data arrays from a single tree.
    Snorm (float, optional): Normalization factor for Stot (default is 100).

    Returns:
    ProcessedData: An instance of ProcessedData containing processed data arrays.
    """
    # Process Nstat
    Nstat = np.log10(tree_data.Nstat)
    
    # Rescale Stot
    def rescale_Stot(x, Snorm):
        return np.log10(x + 1) / math.log10(Snorm + 1)
    
    lg_Stot = rescale_Stot(tree_data.Stot, Snorm)
    
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
    
    # Normalize traces
    traces = tree_data.traces[:, :, :150]
    for j, event in enumerate(traces):
        for i, stat in enumerate(event):
            stat = np.cumsum(stat)
            traces[j][i] = stat / np.max(stat)
    
    return ProcessedData(
        lgE_MC=lgE_MC,
        lgE=lgE,
        Dist=Dist_norm,
        traces=traces,
        ID=tree_data.ID,
        t0=tree_data.t0,
        theta=theta_deg,
        lg_Stot=lg_Stot,
        lg_S1000=lg_S1000,
        azimuth=azimuth_deg,
        Nstat=Nstat
    )


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
        Ene_MC=branches["E_MC"].tolist(),
        Ene=branches["E_SD"].tolist(),
        Dist=branches["Dist"].tolist(),
        traces=branches["traces_vec"].tolist(),
        ID=branches["ID_SD"].tolist(),
        t0=branches["t0"].tolist(),
        theta=branches["theta"].tolist(),
        Stot=branches["Stot"].tolist(),
        S1000=branches["Shsize"].tolist(),
        azimuth=branches["azimuthSP"].tolist(),
        Nstat=branches["Nstat"].tolist()
    )

    return tree_data



def read_config(config_file):
    """
    Read input and output file paths from a JSON configuration file.

    Parameters:
    config_file (str): The path to the JSON configuration file.

    Returns:
    tuple: A tuple containing the input and output file paths:
        - input_file (str): The path to the input ROOT file.
        - output_file (str): The path to the output file.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config.get('input_file'), config.get('output_file')



def extract_particle_name(input_file):
    """
    Extracts the particle name from the input file path.

    Parameters:
    input_file (str): The input file path.

    Returns:
    str: The particle name extracted from the input file path.
    """
    # Extract the directory name containing the input file
    directory = os.path.dirname(input_file)
    
    # Split the directory path by slashes and extract the last part
    parts = directory.split("/")
    particle_name = parts[-1]  # Extract the last part
    
    return particle_name