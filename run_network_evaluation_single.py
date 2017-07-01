###################################################################
# Command line script to analyze network on node sets of interest #
###################################################################

from network_evaluation_tools import network_evaluation_functions as nef
from network_evaluation_tools import data_import_tools as dit
import time
import argparse
import os
import pandas as pd
from multiprocessing import Pool

# Checking valid alpha and p values (Range is 0.0-1.0 exclusive)
def restricted_float(x):
    x = float(x)
    if x <= 0.0 or x >= 1.0:
        raise argparse.ArgumentTypeError("%r not in range (0.0, 1.0) exclusive"%(x,))
    return x

# Checking valid integer values (for all values that must be >0)
def positive_int(x):
    x = int(x)
    if x <= 0:
         raise argparse.ArgumentTypeError("%s must be a positive integer" % x)
    return x

# Valid file path check (Does not check file formatting, but checks if given path exists and is readable)
def valid_infile(in_file):
	if not os.path.isfile(in_file):
		raise argparse.ArgumentTypeError("{0} is not a valid input file path".format(in_file))	
	if os.access(in_file, os.R_OK):
		return in_file
	else:
		raise argparse.ArgumentTypeError("{0} is not a readable input file".format(in_file))

# Valid output directory path check (Checks if the output directory path can be found and written to by removing given filename from full path)
# Note: This uses '/' character for splitting pathnames on Linux and Mac OSX. The character may need to be changed to '\' for Windows executions
def valid_outfile(out_file):
	outdir = '/'.join(out_file.split('/')[:-1])
	if not os.path.isdir(outdir):
		raise argparse.ArgumentTypeError("{0} is not a valid output directory".format(outdir))
	if os.access(outdir, os.W_OK):
		return out_file
	else:
		raise argparse.ArgumentTypeError("{0} is not a writable output directory".format(outdir))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Analyze network performance on ability to aggregate sets of nodes in network space.')
	# Required arguments
	parser.add_argument("network_path", type=valid_infile, 
		help='Path to file of network to be evaluated. File must be 2-column edge list where each line is a gene interaction separated by a common delimiter.')
	parser.add_argument("node_sets_file", type=valid_infile, 
		help='Path to file of node sets. Each line is a list, separated by a common delimiter. The first item in each line will be the name of the node set.')
	parser.add_argument("AUPRCs_save", type=valid_outfile, 
		help='CSV file path of network evaluation result scores (AUPRCs). Must have a writable directory.')	

	# File loading options
	parser.add_argument('-nd', '--net_filedelim', type=str, default='\t', required=False,
		help='Delimiter used in network file between columns. Default is tab white space.')
	parser.add_argument('-sd', '--set_filedelim', type=str, default='\t', required=False,
		help='Delimiter used in node sets file. Default is tab white space.')

	# Network options
	parser.add_argument('-s', '--shuffle', default=False, action="store_true", required=False,
		help='Whether or not to shuffle the network edges (preserving node degree).')	

	# Network propagation options
	parser.add_argument("-a", "--alpha", type=restricted_float, default=None, required=False,
		help='Propagation constant to use in the propagation of node sub-samples over given network. Overrides alpha calculation model if given.')
	parser.add_argument("-m", "--alpha_model_m", type=float, default=-0.17190024, required=False,
		help='Slope of linear alpha calculation model used if no alpha value given.')
	parser.add_argument("-b", "--alpha_model_b", type=float, default=0.7674828, required=False,
		help='Slope of linear alpha calculation model used if no alpha value given.')	

	# Gene set sub-sampling options
	parser.add_argument("-p", "--sample_p", type=restricted_float, default=0.1, required=False,
		help='Sub-sampling percentage for node sets of interest. Default is 0.1.')
	parser.add_argument("-n", "--sub_sample_iter", type=positive_int, default=30, required=False,
		help='Number of times to perform sub-sampling during performance recovery (AUPRC) calculation for each node set. Default is 30.')

	# Other execution options
	parser.add_argument('-v', '--verbose', default=False, action="store_true", required=False,
		help='Verbosity flag for reporting on patient similarity network construction steps.')
	parser.add_argument('-c', '--cores', type=positive_int, default=1, required=False,
		help='Number of cores to be utilized by machine for performance calculation step. NOTE: Each core must have enough memory to store at least network-sized square matrix and given node sets to perform calculations.')

	args = parser.parse_args()

	starttime = time.time()
	# Load network
	network = dit.load_network_file(args.network_path, delimiter=args.net_filedelim, verbose=args.verbose)
	# Load node set
	node_sets = dit.load_node_sets(args.node_sets_file, delimiter=args.set_filedelim, verbose=args.verbose)
	# Shuffle propagation network if needed
	if args.shuffle:
		network = nef.shuffle_network(network, verbose=args.verbose)
	# Calculate network influence matrix
	net_kernel = nef.construct_prop_kernel(network, alpha=args.alpha, verbose=args.verbose)
	# Calculate AUPRC values for each node set
	if args.cores == 1:
		# Calculate AUPRC values for node sets one at a time
		node_set_AUPRCs = {node_set:nef.calculate_AUPRC_serial(net_kernel, args.sample_p, args.sub_sample_iter, node_sets[node_set], verbose=args.verbose) for node_set in node_sets}
	else:
		# Initialize multiple threads for AUPRC analysis of multiple node sets
		initializer_args = [net_kernel]
		pool = Pool(args.cores, nef.parallel_analysis_initializer, initializer_args)
		# Construct parameter list to be passed
		AUPRC_Analysis_params = [[node_set, node_sets[node_set], args.sample_p, args.sub_sample_iter, args.verbose] for node_set in node_sets]
		# Run the AUPRC analysis for each geneset
		AUPRC_results = pool.map(nef.calculate_AUPRC_parallel, AUPRC_Analysis_params)
		pool.close()
		# Construct AUPRC results dictionary
		node_set_AUPRCs = {result[0]:result[1] for result in AUPRC_results}
	AUPRCs_table = pd.DataFrame(pd.Series(node_set_AUPRCs, name='AUPRC'))
	AUPRCs_table.to_csv(args.AUPRCs_save)
	print 'Network AUPRC Analysis complete:', round(time.time()-starttime, 2), 'seconds'


