###############################################
# ---------- Data Import Functions ---------- #
###############################################

import pandas as pd
import networkx as nx
import time
import os

# Load network from file
# Can set delimiter, but default delimiter is tab
# Only will read edges as first two columns, all other columns will be ignored
def load_network_file(network_file_path, delimiter='\t', verbose=False):
	network = nx.read_edgelist(network_file_path, delimiter=delimiter, data=False)
	if verbose:
		print 'Network File Loaded:', network_file_path
	return network

# Get full paths to all networks in directory with a given file name structure:
# e.g. If filename = 'BIND_Symbol.sif', then network_name='BIND', suffix='_Symbol', ext='.sif
def get_networks(wd, suffix=None, file_ext='.sif'):
	network_files = {}
	for fn in os.listdir(wd):
		if suffix==None:
			if fn.endswith(file_ext):
				network_files[fn.split(file_ext)[0]]=wd+fn			
		else:
			if fn.endswith(file_ext) and fn.split(file_ext)[0].endswith(suffix):
				network_files[fn.split(suffix)[0]]=wd+fn
	return network_files

# Companion function with get_networks(), loads all of the network files found in a directory
# Uses the load_network_file() function to load each network, also only imports first two columns, no edge data
# Constructs a dictionary of useful network items for each network in the directory:
#  - Actual networkx object representation of network
#  - List of nodes by name for each network
#  - List of edges by node name for each network
def load_networks(network_file_map, delimiter='\t', verbose=False):
	# Initialize dictionaries
	networks, network_edges, network_nodes = {}, {}, {}
	# Loading network and network properties
	for network_name in network_file_map:
		loadtime = time.time()
		# Load network
		network = load_network_file(network_file_map[network_name], verbose=verbose)
		networks[network_name]=network
		# Construct network node list
		network_nodes[network_name] = network.nodes()
		# Construct network edge list
		network_edges[network_name] = network.edges()
	if verbose:
		print 'All given network files loaded'
	# Return data structure
	return networks, network_edges, network_nodes

# Convert and save MAF from Broad Firehose
# Can produce 2 types of filetypes: 'matrix' or 'list', matrix is a full samples-by-genes binary csv, 'list' is a sparse representaiton of 'matrix'
# This is a conversion tool, so the result must be saved (most tools will require a path to a processed MAF file and load it separately)
def process_TCGA_MAF(maf_file, save_path, filetype='matrix', verbose=False):
	loadtime = time.time()
	# Load MAF File
	TCGA_MAF = pd.read_csv(maf_file,sep='\t',low_memory=False)
	# Get all patient somatic mutation (sm) pairs from MAF file
	TCGA_sm = TCGA_MAF.groupby(['Tumor_Sample_Barcode', 'Hugo_Symbol']).size()
	# Turn somatic mutation data into binary matrix
	TCGA_sm_mat = TCGA_sm.unstack().fillna(0)
	TCGA_sm_mat = (TCGA_sm_mat>0).astype(int)
	# Trim TCGA barcodes
	TCGA_sm_mat.index = [pat[:12] for pat in TCGA_sm_mat.index]
	# Filter samples with duplicate IDs
	non_dup_IDs = list(TCGA_sm_mat.index.value_counts().index[TCGA_sm_mat.index.value_counts()==1])
	dup_IDs = list(TCGA_sm_mat.index.value_counts().index[TCGA_sm_mat.index.value_counts()>1])
	# Save file as binary matrix or sparse list
	if filetype=='list':
		# Now try to construct two-column/sparse representation of binary sm data
		# Get list of all patient somatic mutations
		index_list = list(TCGA_sm.index)
		# Filter list of patient somatic mutations of duplicate patient barcodes
		index_list_filt = [i for i in index_list if not any([True if barcode in i[0] else False for barcode in dup_IDs])]
		# Save patient somatic mutations list to file
		f = open(save_path, 'w')
		for sm in index_list_filt:
			f.write(sm[0][:12]+'\t'+sm[1]+'\n')
		f.close()
		if verbose:
			print 'Binary somatic mutations list saved'
	else:
		# Save non-duplicate patients' binary TCGA somatic mutation matrix to csv
		TCGA_sm_mat_filt = TCGA_sm_mat.ix[non_dup_IDs]
		# Remove all genes that have no more mutations after patient filtering
		empty_cols = [col for col in TCGA_sm_mat_filt.columns if not all(TCGA_sm_mat_filt[col]==0)]
		TCGA_sm_mat_filt2 = TCGA_sm_mat_filt[empty_cols]
		TCGA_sm_mat_filt2.to_csv(save_path)
		if verbose:
			print 'Binary somatic mutation matrix saved'
	if verbose:
		print 'MAF file processed:', maf_file, round(time.time()-loadtime, 2), 'seconds.'
	return

# Load binary mutation data with 2 file types (filetype= 'matrix' or 'list')
# filetype=='matrix' is a csv or tsv style matrix with row and column headers, rows are samples/patients, columns are genes
# filetype=='list' is a 2 columns text file separated by the delimiter where 1st column is sample/patient, 2nd column is one gene mutated in that patient
# Line example in 'list' file: 'Patient ID','Gene Mutated'
def load_binary_mutation_data(filename, filetype='matrix', delimiter=',', verbose=False):
	if filetype=='list':
		f = open(filename)
		binary_mat_lines = f.read().splitlines()
		binary_mat_data = [(line.split('\t')[0], line.split('\t')[1]) for line in binary_mat_lines]
		binary_mat_index = pd.MultiIndex.from_tuples(binary_mat_data, names=['Tumor_Sample_Barcode', 'Hugo_Symbol'])
		binary_mat_2col = pd.DataFrame(1, index=binary_mat_index, columns=[0])[0]
		binary_mat = binary_mat_2col.unstack().fillna(0)
	else:
		binary_mat = pd.read_csv(filename, delimiter=delimiter, index_col=0).astype(int)
	if verbose:
	   print 'Binary Mutation Matrix Loaded:', filename
	return binary_mat

# Concatinate multiple mutation matrices together
# All file type structures and delimiters must be the same (see load_binary_mutation_matrix()) across all files
def concat_binary_mutation_matrices(filename_list, filetype='matrix', delimiter=',', verbose=False, save_path=None):
	binary_mat_list = [load_binary_mutation_data(fn, filetype=filetype, delimiter=delimiter, verbose=verbose) for fn in filename_list]  
	binary_mat_concat = pd.concat(binary_mat_list).fillna(0)
	if verbose:
		print 'All binary mutation matrices loaded and concatenated'
	if save_path==None:
		return binary_mat_concat
	else:
		binary_mat_concat.to_csv(save_path)
		return binary_mat_concat

# Construct dictionary of node sets from input text file to perform AUPRC analysis on for network of interest
# File format: Each line is a delimited list with the first item in the list is the name of the node set
# All other nodes in the list follow the node set name
def load_node_sets(node_set_file, delimiter='\t', verbose=False):
	f = open(node_set_file)
	node_set_lines = f.read().splitlines()
	node_set_lines_split = [line.split(delimiter) for line in node_set_lines]
	f.close()
	node_sets = {node_set[0]:set(node_set[1:]) for node_set in node_set_lines_split}
	if verbose:
		print 'Node cohorts loaded:', node_set_file
	return node_sets