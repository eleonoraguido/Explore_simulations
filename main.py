import sys
import utils, plots, create_dataset

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    config_type = utils.read_config_type(config_file)

    if config_type == "plot":
        print("A single .root file is given as input. Plots will be created to explore it.")
        input_file, output_file, theta_cut = utils.read_config_plot(config_file)
        print("\tInput file is "+input_file)
        print("\tOutput directory is "+output_file)
        data = utils.read_tree(input_file) #read the tree content
        proc_data = utils.process_data(data, theta_cut)   # modify the tree content
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
        input_file1, input_file2, output_dir, dataset_name, theta_cut = utils.read_config_dataset(config_file)
        print("File 1 is ",input_file1)
        print("File 2 is ",input_file2)
        data1 = utils.read_tree(input_file1) #read the tree content
        data2 = utils.read_tree(input_file2) #read the tree content
        proc_data1 = utils.process_data(data1, theta_cut, 1)   # modify the tree content
        proc_data2 = utils.process_data(data2, theta_cut, 0)   # modify the tree content
        preprocessed_data = []
        preprocessed_data.append(proc_data1)
        preprocessed_data.append(proc_data2)

        create_dataset.merge_and_shuffle(output_dir,dataset_name,preprocessed_data)
        create_dataset.load_npz_file(output_dir, dataset_name)   #to check if it worked

    elif config_type == "onepart_dataset":
        print("One .root file is given as input.")
        input_file, output_dir, dataset_name, theta_cut, label = utils.read_config_onepart_dataset(config_file)
        print("File is ",input_file)
        print("The data set will be created and the events will be labelled as "+ str(label)+".")
        data = utils.read_tree(input_file) #read the tree content
        proc_data = utils.process_data(data, theta_cut, label)   # modify the tree content
        preprocessed_data = []
        preprocessed_data.append(proc_data)

        create_dataset.merge_and_shuffle(output_dir,dataset_name,preprocessed_data)
        create_dataset.load_npz_file(output_dir, dataset_name)   #to check if it worked

        
    else:
        print("Unsupported config file type.")
        sys.exit(1)

    
    


if __name__ == "__main__":
    main()