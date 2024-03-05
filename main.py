import sys
import utils, plots

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    input_file, output_file = utils.read_config(config_file)
    
    data = utils.read_tree(input_file) #read the tree content
    proc_data = utils.process_data(data)   # modify the tree content

    particle_name = utils.extract_particle_name(input_file)

    #Plots
    plots.plot_energy_distributions(output_file, proc_data.lgE_MC, proc_data.lgE, particle_name)
    plots.plot_S1000_vs_theta(output_file, proc_data.theta, proc_data.lg_S1000, particle_name)
    plots.plot_nstat_distribution(output_file, proc_data.Nstat, particle_name)
    plots.plot_hottest_stations(output_file, proc_data.Dist, particle_name)
    plots.plot_theta_distribution(proc_data.theta, output_file, particle_name)
    plots.plot_traces(data.traces, proc_data.traces, output_file, particle_name)


if __name__ == "__main__":
    main()