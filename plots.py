import os
import matplotlib.pyplot as plt

def plot_energy_distributions(output_dir, lgE2_MC, lgE2, particle_name):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
    
    _, bins, _ = ax[1].hist(lgE2, range=(17.7, 19.5), bins=18, rwidth=0.95, color="orange", ec="black", zorder=4, label=particle_name)
    ax[1].set_title("$E_{SD}$ distributions", fontsize=16)
    ax[1].grid(zorder=0)
    ax[1].set_ylabel('#', fontsize=13)
    ax[1].set_xlabel('$log_{10}(E/eV)$', fontsize=13)

    ax[0].hist(lgE2_MC, bins, rwidth=0.95, color="orange", ec="black", zorder=5, label=particle_name, linestyle=('solid')) 
    ax[0].set_title("$E_{MC}$ distributions", fontsize=16)
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