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
    bin_edges = np.linspace(0, 20, num=21)  # 21 edges for 20 bins
    _, bins, _ = ax.hist(10**Nstat, bins=bin_edges, rwidth=0.95, color="lightblue", ec="blue", zorder=4, label='All')
    ax.set_title("Number of Triggered Stations", fontsize=16)
    ax.grid(zorder=0)
    ax.set_ylabel('Number of Events', fontsize=13)
    ax.set_xlabel('$\log_{10}(N_{stat})$', fontsize=13)

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