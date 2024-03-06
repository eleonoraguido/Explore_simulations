import sys
import utils, plots

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    config_type = utils.read_config_type(config_file)

    if config_type == "plot":
        print("A single .root file is given as input. Plots will be created to explore it.")
        input_file, output_file = utils.read_config_plot(config_file)
        data = utils.read_tree(input_file) #read the tree content
        proc_data = utils.process_data(data)   # modify the tree content
        particle_name = utils.extract_particle_name(input_file)
        #Plots
        plots.plot_energy_distributions(output_file, proc_data.lgE_MC, proc_data.lgE, particle_name)
        plots.plot_S1000_vs_theta(output_file, proc_data.theta, proc_data.lg_S1000, particle_name)
        plots.plot_nstat_distribution(output_file, proc_data.Nstat, particle_name)
        plots.plot_hottest_stations(output_file, proc_data.Dist, particle_name)
        plots.plot_theta_distribution(proc_data.theta, output_file, particle_name)
        plots.plot_traces(data.traces, proc_data.traces_cum, output_file, particle_name)
        plots.plot_Stot_for3stations(output_file, proc_data.lg_Stot, particle_name)        
    elif config_type == "dataset":
        print("Two .root files are given as input.")
        print("The data set with signal and background events will be created")
        input_file1, input_file2, output_dir = utils.read_config_dataset(config_file)
        print(input_file1)
        print(input_file2)
        print(output_dir)
    else:
        print("Unsupported config file type.")
        sys.exit(1)

    
    


if __name__ == "__main__":
    main()