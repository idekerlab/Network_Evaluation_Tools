##################################################################
# Command line script to analyze network on node sets of interest#
##################################################################

from network_evaluation_tools import network_evaluation_functions as nef
import argparse

# Checking valid alpha and p values (Range is 0.0-1.0 exclusive)
def restricted_float(x):
    x = float(x)
    if x <= 0.0 or x >= 1.0:
        raise argparse.ArgumentTypeError("%r not in range (0.0, 1.0) exclusive"%(x,))
    return x

# Checking valid integer values (for all values that must be >0)
def positive_int(x):
    x = int(value)
    if x <= 0:
         raise argparse.ArgumentTypeError("%s must be a positive integer" % value)
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
	outdir = '/'.join(outfile.split('/')[:-1])
	if not os.path.isdir(outdir):
		raise argparse.ArgumentTypeError("{0} is not a valid output directory".format(outdir))
	if os.access(outdir, os.W_OK):
		return out_file
	else:
		raise argparse.ArgumentTypeError("{0} is not a writable output directory".format(outdir))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Analyze network performance on ability to aggregate sets of nodes in network space.')
	parser.add_argument("network_path", type=valid_infile, 
		help='Path to file of network to be evaluated. File must be 2-column edge list where each line is a gene interaction separated by a common delimiter.')
	parser.add_argument("node_sets_file", type=valid_infile, 
		help='Path to file of node sets. Each line is a list, separated by a common delimiter. The first item in each line will be the name of the node set.')
	parser.add_argument('AUPRCs_save', type=valid_outfile, 
		help='CSV file path of network evaluation result scores (AUPRCs). Must have a writable directory.')	
	parser.add_argument("-p", "--sample_p", type=restricted_float, default=0.1, required=False,
		help='Sub-sampling percentage for node sets of interest. Default is 0.1.')
	parser.add_argument("-n", "--sub_sample_iter", type=positive_int, default=30, required=False,
		help='Number of times to perform sub-sampling during performance recovery (AUPRC) calculation for each node set. Default is 30.')
	parser.add_argument("-i", "--null_iter", type=positive_int, default=30, required=False,
		help='Number of times to perform degree-preserved shuffling of network to construct performance value null distribution. Default is 30.')
	parser.add_argument('-nd', '--net_filedelim', type=str, default='\t', required=False,
		help='Delimiter used in network file between columns. Default is tab white space.')
	parser.add_argument('-d', '--set_filedelim', type=str, default='\t', required=False,
		help='Delimiter used in node sets file. Default is tab white space.')
	parser.add_argument("-a", "--alpha", type=restricted_float, default=None, required=False,
		help='Propagation constant to use in the propagation of node sub-samples over given network. Overrides alpha calculation model if given.')
	parser.add_argument("-m", "--alpha_model_m", type=float, default=-0.17190024, required=False,
		help='Slope of linear alpha calculation model used if no alpha value given.')
	parser.add_argument("-b", "--alpha_model_b", type=float, default=0.7674828, required=False,
		help='Slope of linear alpha calculation model used if no alpha value given.')
	parser.add_argument('-v', '--verbose', default=False, action="store_true", required=False,
		help='Verbosity flag for reporting on patient similarity network construction steps.')
	parser.add_argument('-c', '--cores', default=positive_int, default=1, required=False,
		help='Number of cores to be utilized by machine for performance calculation step. NOTE: Each core must have enough memory to store at least network-sized square matrix and given node sets to perform calculations.')
	parser.add_argument('-z', '--calculate_Z', default=False, action="store_true", required=False,
		help='Whether or not to calculate Z-scores for network performance. If True, this flag will require --save_AUPRCs and --save_null_AUPRCs paths.')
	parser.add_argument('-sn', '--null_AUPRCs_save', type=valid_outfile, default=None, required=False,
		help='CSV file path of where to save null network evaluation results. Required if calculating performance z-scores (-z flag).')
	parser.add_argument('-sz', '--ZScores_save', type=valid_outfile, default=None, required=False,
		help='CSV file path of where to save network evaluation results as z-scores. Required if calculating performance z-scores (-z flag).')
	args = parser.parse_args()
	# If Z-scores are to be calculated, check if required save parameters are given
	if args.calculate_Z:
		# First check if required output paths are given
		if args.null_AUPRCs_save is None:
			parser.error('Save path required for null network performance results (-sn or --null_AUPRCs_save')
		if args.ZScores_save is None:
			parser.error('Save path required for z-scored network performance results (-sz or --ZScores_save')


	# Perform AUPRC Calculation on Patient similarity network and given cohort sets
	AUPRCs = AUPRC_Analysis(network_file, node_set_file, sample_p, sub_sample_iterations, 
		alpha=args.alpha, m=args.alpha_model_m, b=args.alpha_model_b, 
		net_delim=args.net_filedelim, set_delim=args.set_filedelim, 
		cores=args.cores, verbose=args.verbose, save_path=args.AUPRCs_save)

	# If a Z-score calculation is desired, then we must calculate null AUPRCs, then Z-Score
	if args.calculate_Z:
		# Calculate null AUPRCs
		null_AUPRCs = null_AUPRC_Analysis(network_file, node_set_file, sample_p, sub_sample_iterations, null_iterations, 
			alpha=args.alpha, m=args.alpha_model_m, b=args.alpha_model_b, 
			net_delim=args.net_filedelim, set_delim=args.set_filedelim, 
			cores=args.cores, verbose=args.verbose, save_path=args.null_AUPRCs_save)
		# Calculate Z-Score
		ZScores = AUPRC_Analysis_with_ZNorm(args.AUPRCs_save, args.null_AUPRCs_save, verbose=args.verbose, save_path=args.ZScores_save)