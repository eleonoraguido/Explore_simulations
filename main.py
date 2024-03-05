import sys
import utils, plots

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    input_file, output_file = utils.read_config(config_file)
    
    data = utils.read_tree(input_file) #read the tree content
    proc_data = utils.process_data(data)   #modify the tree content

    particle_name = utils.extract_particle_name(input_file)

    plots.plot_energy_distributions(output_file, proc_data.lgE_MC, proc_data.lgE, particle_name)


if __name__ == "__main__":
    main()