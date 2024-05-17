"""
Author: Eleonora Guido
Last modification date: 05.2024
Photon search with a CNN
"""


import os
import matplotlib.pyplot as plt
import numpy as np

def plot_energy_distributions(output_dir, lgE_MC, lgE, particle_name):
    """
    Plots energy distributions for Monte Carlo and reconstructed energy, 
    along with a scatter plot comparing the two energies.

    Parameters:
    output_dir (str): Directory where the plot will be saved.
    lgE2_MC (np.ndarray): Processed MC energy.
    lgE2 (np.ndarray): Processed reconstructed energy.
    particle_name (str): Name of the particle being analyzed.

    Returns:
    None
    """
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
    
    _, bins, _ = ax[1].hist(lgE, range=(17.7, 19.5), bins=18, rwidth=0.95, color="orange", ec="black", zorder=4, label=particle_name)
    ax[1].set_title("$E_{SD}$ distribution", fontsize=16)
    ax[1].grid(zorder=0)
    ax[1].set_ylabel('#', fontsize=13)
    ax[1].set_xlabel('$log_{10}(E/eV)$', fontsize=13)

    ax[0].hist(lgE_MC, bins, rwidth=0.95, color="orange", ec="black", zorder=5, label=particle_name, linestyle=('solid')) 
    ax[0].set_title("$E_{MC}$ distribution", fontsize=16)
    ax[0].grid(zorder=0)
    ax[0].set_ylabel('#', fontsize=13)
    ax[0].set_xlabel('$log_{10}(E/eV)$', fontsize=13)
    
    plt.tight_layout()
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot in the output directory with the provided filename
    filename = f"{particle_name}_energy_distribution.pdf"  # Adjust the filename as needed
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path)

    # Plot scatter plot comparing MC and reconstructed energies
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    lims = [
        np.min([np.min([lgE_MC, lgE])]),  # min of both axes
        np.max([np.max([lgE_MC, lgE])]),  # max of both axes
    ]
    ax2.scatter(lgE_MC, lgE, s=1, color='darkgreen') 
    ax2.grid(zorder=0)
    ax2.plot(lims, lims, color='k')
    ax2.set_ylabel('$\log_{10}(E_{SD}/\mathrm{eV})$', fontsize=13)
    ax2.set_xlabel('$\log_{10}(E_{MC}/\mathrm{eV})$', fontsize=13)
    ax2.set_title('Comparison of MC and Reconstructed Energies', fontsize=15)

    plt.tight_layout()
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot in the output directory with the provided filename
    filename = f"{particle_name}_energy_comparison.pdf"  # Adjust the filename as needed
    output_path = os.path.join(output_dir, filename)
    fig2.savefig(output_path)




def plot_S1000_vs_theta(output_dir, theta, lg_S1000, particle_name):
    """
    Plots S1000 vs. theta and saves the plot.

    Parameters:
    output_dir (str): Directory where the plot will be saved.
    theta (np.ndarray): Theta values.
    lg_S1000 (np.ndarray): S1000 values.
    particle_name (str): Name of the particle being analyzed.

    Returns:
    None
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.set_title(particle_name+" distribution", fontsize=15)
    _, binsx, binsy, im = ax.hist2d(theta, lg_S1000, bins=100, cmap='cividis')
    ax.set_ylabel('$log_{10}(S1000/\mathrm{VEM})$', fontsize=13)
    ax.set_xlabel('$\\theta$ (°)', fontsize=13)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot in the output directory with the provided filename
    filename = f"{particle_name}_S1000_vs_theta.pdf"  # Adjust the filename as needed
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, bbox_inches='tight')




def plot_nstat_distribution(output_dir, Nstat, particle_name):
    """
    Plots the distribution of the number of triggered stations for a single particle.

    Parameters:
    output_dir (str): Directory where the plot will be saved.
    Nstat (np.ndarray): Array containing the number of triggered stations for the particle.
    particle_name (str): Name of the particle being analyzed.

    Returns:
    None
    """
    fig, ax = plt.subplots()
    Nstat_val = 10**Nstat
    bin_edges = np.linspace(0, 20, num=21)  # 21 edges for 20 bins
    _, bins, _ = ax.hist(Nstat_val, bins=bin_edges, rwidth=0.95, color="lightblue", ec="blue", zorder=4, label='All')
    ax.set_title("Number of Triggered Stations", fontsize=16)
    ax.grid(zorder=0)
    ax.set_ylabel('Number of Events', fontsize=13)
    ax.set_xlabel('$N_{stat}$', fontsize=13)
    ax.set_xticks(np.arange(min(bins), max(bins)+1, 1))
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot in the output directory with the provided filename
    filename = f"{particle_name}_Nstat_distribution.pdf"  # Adjust the filename as needed
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path)



def plot_hottest_stations(output_dir, Dist_norm, particle_name):
    """
    Plot the distance distribution of the three hottest stations from the shower core.

    Parameters:
    Dist_norm (np.ndarray): Array containing normalized distances of the hottest stations.

    Returns:
    None
    """
    plt.figure()
    plt.hist(Dist_norm[:, 0], alpha=0.7, ec="blue", zorder=4, label='Hottest station')
    plt.hist(Dist_norm[:, 1], alpha=0.5, ec="orange", zorder=5, label='Second-hottest station')
    _, bins, _ = plt.hist(Dist_norm[:, 2], alpha=0.2, ec="green", zorder=8, label='Third-hottest station')
    
    plt.title("Distance of the three hottest stations from the shower core")
    plt.grid(zorder=0)
    plt.ylabel('#')
    plt.xlabel('$\\tilde{d}$')
    plt.legend()

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot in the output directory with the provided filename
    filename = f"{particle_name}_stations_distance.pdf"  # Adjust the filename as needed
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)


def plot_Stot_for3stations(output_dir, Stot, particle_name):
    """
    Plots the total signal distribution for three stations of a given particle.

    Parameters:
    output_dir (str): Directory where the plot will be saved.
    Stot (np.ndarray): Array containing the total signal values for all stations.
    particle_name (str): Name of the particle being analyzed.

    Returns:
    None
    """
    n_stat = 3 
    for num_stat in range(0,n_stat):    #loop over the nth hottest stations
        fig, ax = plt.subplots()
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.6)
        Stot_stat = np.array([Stot_val for Stot_val in Stot])[:,num_stat]
        _, bins, _ = ax.hist(Stot_stat,range=(np.nanmin(Stot_stat), np.nanmax(Stot_stat)), rwidth = 0.95, ec=(210/255,105/255,30/255,1), facecolor=(255/255,228/255,196/255,0.6), zorder=6, label=particle_name)
        ax.set_title("Total signal for station {0}".format(num_stat+1), fontsize=15)
        ax.set_ylabel('#', fontsize=13)
        ax.set_xlabel('$S_{tot}^{norm}$', fontsize=12)
        ax.grid(zorder=0)
        ax.text(0.3, 0.65, particle_name+' : $\mu$={0}, $\sigma={1}$'.format(round(Stot_stat.mean(),2), round(Stot_stat.std(),2)), horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, zorder=7, bbox=props, fontsize=12) 
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the plot in the output directory with the provided filename
        filename = f"{particle_name}"  # Adjust the filename as needed
        output_path = os.path.join(output_dir, filename)
        fig.savefig(output_path+'_Stot_stat_{0}.pdf'.format(num_stat), bbox_inches='tight')


def plot_theta_distribution(theta, output_dir, particle_name):
    """
    Plot the zenith angle distribution for a single particle.

    Parameters:
    theta (np.ndarray): Array containing zenith angles.
    output_dir (str): Directory where the plot will be saved.
    particle_name (str): Name of the particle being analyzed.

    Returns:
    None
    """
    fig, ax = plt.subplots()
    ax.hist(np.radians(theta),bins=75, range=(0, np.radians(75)), rwidth=0.95, color="lightblue", ec="blue", zorder=4)

    def deg2rad(x):
        return x * np.pi / 180
    
    def rad2deg(x):
        return x * 180 / np.pi
    
    secax = ax.secondary_xaxis('top', functions=(rad2deg, deg2rad))
    secax.set_xlabel('$\\theta$ (°)')
    ax.grid(zorder=0)
    ax.set_title("Zenith angle distribution")
    ax.set_ylabel('#')
    ax.set_xlabel('$\\theta$ (rad)')
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot in the output directory with the provided filename
    filename = f"{particle_name}_theta_distribution.pdf"  # Adjust the filename as needed
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, bbox_inches='tight')



def plot_traces(traces, cum_traces, output_dir, particle_name):
    """
    Plot the traces for a single particle.

    Parameters:
    traces (np.ndarray): Array containing traces for the particle.
    cum_traces (np.ndarray): Array containing cumulative traces for the particle.
    output_dir (str): Directory where the plot will be saved.
    particle_name (str): Name of the particle being analyzed.

    Returns:
    None
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot traces for the first particle
    selected_indices = np.random.choice(np.arange(cum_traces.shape[0]), size=100, replace=False)
    for idx in selected_indices:
        axes[0].plot(traces[idx, 0, :], label=f'Trace {idx}')
    axes[0].set_title("Traces for " + particle_name+"s")
    axes[0].set_ylabel('VEM')
    axes[0].set_xlabel('Time bin')

    # Plot traces for the second particle
    for idx in selected_indices:
        axes[1].plot(cum_traces[idx, 0, :], label=f'Trace {idx}')
    axes[1].set_title("Cumulative traces for " + particle_name +"s")
    axes[1].set_ylabel('VEM')
    axes[1].set_xlabel('Time bin')
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the plot in the output directory with the provided filename
    filename = f"{particle_name}_traces.pdf"  # Adjust the filename as needed
    output_path = os.path.join(output_dir, filename)
    fig.savefig(output_path, bbox_inches='tight')