##############################################################
# Command line script to construct patient simlarity network #
##############################################################

from network_evaluation_tools import patient_similarity_network as psn
import argparse
import os

# Range check for valid alpha values only
def alpha_range(x):
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
	parser = argparse.ArgumentParser(description='Construct a patient similarity network given a molecular network and binary mutation matrix.')
	parser.add_argument("network_path", type=valid_infile, 
		help='Path to molecular network file. File must be 2-column edge list where each line is a gene interaction separated by a common delimiter.')
	parser.add_argument("mutation_data", type=valid_infile, 
		help='Path to binary mutation matrix file. May be a csv or 2-column list where each line is a sample and the gene mutated separated by a common delimiter.')
	parser.add_argument('network_save_file', type=valid_outfile,
		help='File path of where to 2-column edge list file of constructed patient similarity network. Must have be in a writable directory.')	
	parser.add_argument('-nd', '--net_filedelim', type=str, default='\t', required=False,
		help='Delimiter used in network file between columns. Default is tab white space.')
	parser.add_argument('-mf', '--mut_filetype', type=str, default='matrix', choices=['matrix', 'list'], required=False,
		help='File structure of binary mutation data. 2 options: "matrix" (e.g. csv or tsv) or "list" (2-column list).')
	parser.add_argument('-md', '--mut_filedelim', type=str, default='\t', required=False,
		help='Delimiter used in binary mutation file. Default is tab white space.')
	parser.add_argument('-s', '--similarity', type=str, default='spearman', required=False,
		help='The measure of similarity used to compare network-propagated profiles of patients to construct the patient similarity network. Default is spearman correlation.')
	parser.add_argument("-a", "--alpha", type=alpha_range, default=None, required=False,
		help='Propagation constant to use in the propagation of mutations over molecular network. Range is 0.0-1.0 exclusive. Overrides alpha calculation model if given.')
	parser.add_argument("-m", "--alpha_model_m", type=float, default=-0.17190024, required=False,
		help='Slope of linear alpha calculation model used if no alpha value given.')
	parser.add_argument("-b", "--alpha_model_b", type=float, default=0.7674828, required=False,
		help='Slope of linear alpha calculation model used if no alpha value given.')
	parser.add_argument('-mm', '--min_muts', type=positive_int, default=1, required=False,
		help='Minimum number of mutations for a sample on the network to be considered in the patient similarity network.')
	parser.add_argument('-v', '--verbose', default=False, action="store_true", required=False,
		help='Verbosity flag for reporting on patient similarity network construction steps.')
	parser.add_argument('-sp', '--save_propagation', type=valid_outfile, default=None, required=False,
		help='File path of where to save sample propagation results. No path given as default, automatically saves csv file if file path given.')
	parser.add_argument('-ss', '--save_similarity', type=valid_outfile, default=None, required=False,
		help='File path of where to save sample-by-sample similarity measurements. No path given as default, automatically saves csv tile if file path given.')
	args = parser.parse_args()

	# Construct patient similarity network
	PSN = psn.PSN_Constructor(args.network_path, args.mutation_data, 
		net_delim=args.net_filedelim, mut_mat_filetype=args.mut_filetype, mut_mat_delim=args.mut_filedelim, 
		alpha=args.alpha, m=args.alpha_model_m, b=args.alpha_model_b, 
		min_mut=args.min_muts, verbose=args.verbose, similarity=args.similarity,
		save_propagation=args.save_propagation, save_similarity=args.save_similarity, save_network=args.network_save_file)
