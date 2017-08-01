#####################################################################
# ---------- Node Set-Based Network Evaluation Functions ---------- #
#####################################################################
from multiprocessing import Pool
from network_evaluation_tools import data_import_tools as dit
from network_evaluation_tools import network_propagation as prop
import networkx as nx
import numpy as np
import os
import random
import scipy.stats as stats
import sklearn.metrics as metrics
import pandas as pd
import time

# Shuffle network in degree-preserving manner
# Input: network - networkx formatted network
# For large networks this can be slow: may need to be sped up to prevent bottlenecking
def shuffle_network(network, max_tries_n=10, verbose=False):
	# Shuffle Network
	shuff_time = time.time()
	edge_len=len(network.edges())
	shuff_net=network.copy()
	try:
		nx.double_edge_swap(shuff_net, nswap=edge_len, max_tries=edge_len*max_tries_n)
	except:
		if verbose:
			print 'Note: Maximum number of swap attempts ('+repr(edge_len*max_tries_n)+') exceeded before desired swaps achieved ('+repr(edge_len)+').'
	if verbose:
		# Evaluate Network Similarity
		shared_edges = len(set(network.edges()).intersection(set(shuff_net.edges())))
		print 'Network shuffled:', time.time()-shuff_time, 'seconds. Edge similarity:', shared_edges/float(edge_len)
	return shuff_net

# Calculate optimal sub-sampling proportion for test/train
# Input: NetworkX object and dictionary of {geneset name:list of genes}
def calculate_p(network, nodesets, m=-0.18887257, b=0.64897403):
	network_nodes = [str(gene) for gene in network.nodes()]
	nodesets_p = {}
	for nodeset in nodesets:
		nodesets_coverage = len([node for node in nodesets[nodeset] if node in network_nodes])
		nodesets_p[nodeset] = round(m*np.log10(nodesets_coverage)+b, 4)
	return nodesets_p

# Construct influence matrix of each network node propagated across network to use as kernel in AUPRC analysis
# Input: NetowkrkX object. No propagation constant or alpha model required, can be calculated
def construct_prop_kernel(network, alpha=None, m=-0.17190024, b=0.7674828, verbose=False, save_path=None):
	network_Fo = pd.DataFrame(data=np.identity(len(network.nodes())), index=network.nodes(), columns=network.nodes())
	if alpha is None:
		alpha_val = prop.calculate_alpha(network, m=m, b=b)
	else:
		alpha_val = alpha
	network_Fn = prop.closed_form_network_propagation(network, network_Fo, alpha_val, verbose=verbose)
	network_Fn = network_Fn.ix[network_Fn.columns]
	if verbose:
		print 'Propagated network kernel constructed'
	if save_path is not None:
		if save_path.endswith('.hdf'):
			network_Fn.to_hdf(save_path, key='Kernel', mode='w')
		else:
			network_Fn.to_csv(save_path)
	return network_Fn

# Global variable initialization function for small network AUPRC calculations
def global_var_initializer(global_net_kernel):
	global kernel
	kernel = global_net_kernel

# Calculate AUPRC of a single node set's recovery for small networks (<250k edges)
# This method is faster for smaller networks, but still has a relatively large memory footprint
# The parallel setup for this situation requires passing the network kernel to each individual thread
def calculate_small_network_AUPRC(params):
	node_set_name, node_set, p, n, verbose = params[0], params[1], params[2], params[3], params[4]
	runtime = time.time()
	intersect = [nodes for nodes in node_set if nodes in kernel.index]
	AUPRCs = []
	sample_size = int(round(p*len(intersect)))
	for i in range(n):																				# Number of times to run the sampling
		sample = random.sample(intersect, sample_size)													# get node set sample
		intersect_non_sample = [node for node in intersect if node not in sample]					   	# nodes in intersect not in sample
		kernel_non_sample = list(kernel.index[~kernel.index.isin(sample)])							 	# nodes in network not in sample
		kernel_sample_sum = kernel.ix[sample][kernel_non_sample].sum().sort_values(ascending=False)		# summed prop value for all nodes
		y_actual = pd.Series(0, index=kernel_sample_sum.index, dtype=int)								# nodes sorted by mean prop value
		y_actual.ix[intersect_non_sample]+=1															# which nodes in sorted list are in intersect_non_sample
		intersect_non_sample_sorted = y_actual[y_actual==1].index									   	# intersect_non_sample sorted
		TP, FN = 0, len(intersect_non_sample_sorted)													# initialize precision and recall curves
		precision, recall = [1], [0]																	# initialize true positives and false negatives
		for node in intersect_non_sample_sorted:														# Slide down sorted nodes by summed prop value by nodes that are in intersect_non_sample
			TP += 1.0									   													# Calculate true positives found at this point in list
			FN -= 1.0																					   	# Calculate false negatives found at this point in list
			precision.append(TP/float(y_actual.ix[:node].shape[0]))										 	# Calculate precision ( TP / TP+FP ) and add point to curve
			recall.append(TP/float(TP+FN))																  	# Calculate recall ( TP / TP+FN ) and add point to curve
		AUPRCs.append(metrics.auc(recall, precision))												   		# Calculate Area Under Precision-Recall Curve (AUPRC)
	if verbose:
		print 'AUPRC Analysis for given node set', '('+repr(len(intersect))+' nodes in network) complete:', round(time.time()-runtime, 2), 'seconds.'
	return [node_set_name, np.mean(AUPRCs)]

# Caclulate AUPRC of a single node set's recovery for large networks (>=250k edges)
# This method is slower than the small network case, as well as forces the memory footprint to be too large
# The parallel setup for this situation requries 
def calculate_large_network_AUPRC(params):
	geneset, intersect_non_sample_sorted, P_totals, verbose = params[0], params[1], params[2], params[3]
	runtime = time.time()
	TP, FN = 0, len(intersect_non_sample_sorted)	# initialize true positives and false negatives
	precision, recall = [1], [0]					# initialize precision and recall curves
	for node in intersect_non_sample_sorted:		# Step down sorted nodes by summed prop value by nodes that are in intersect_non_sample
		TP += 1.0										# Calculate true positives found at this point in list
		FN -= 1.0										# Calculate false negatives found at this point in list
		precision.append(TP/float(P_totals[node]))		# Calculate precision ( TP / TP+FP ) and add point to curve
		recall.append(TP/float(TP+FN))					# Calculate recall ( TP / TP+FN ) and add point to curve
	AUPRC = metrics.auc(recall, precision)				# Calculate Area Under Precision-Recall Curve (AUPRC)
	if verbose:
		print 'AUPRC Analysis for given node set:', geneset, 'complete:', round(time.time()-runtime, 2), 'seconds.'	
	return [geneset, AUPRC]

# Wrapper to calculate AUPRC of multiple node sets' recovery for small networks (<250k edges)
def small_network_AUPRC_wrapper(net_kernel, genesets, genesets_p, n=30, cores=1, verbose=True):
	# Construct params list
	AUPRC_Analysis_params = [[geneset, genesets[geneset], genesets_p[geneset], n, verbose] for geneset in genesets]
	# Determine parallel calculation status
	if cores == 1:
		# Set network kernel
		global_var_initializer(net_kernel)
		# Calculate AUPRC values for all gene sets
		AUPRC_results = []
		for params_list in AUPRC_Analysis_params:
			AUPRC_results.append(calculate_small_network_AUPRC(params_list))
	else:
		# Initialize worker pool
		pool = Pool(cores, global_var_initializer, [net_kernel])
		# Run the AUPRC analysis for each geneset
		AUPRC_results = pool.map(calculate_small_network_AUPRC, AUPRC_Analysis_params)
		# Close worker pool
		pool.close()
	# Construct AUPRC results
	geneset_AUPRCs = {result[0]:result[1] for result in AUPRC_results}		
	AUPRCs_table = pd.Series(geneset_AUPRCs, name='AUPRC')
	return AUPRCs_table

# Wrapper to calculate AUPRC of multiple node sets' recovery for large networks (>=250k edges)
def large_network_AUPRC_wrapper(net_kernel, genesets, genesets_p, n=30,cores=1, verbose=True):
	starttime = time.time()
	# Construct binary gene set sub-sample matrix
	geneset_list = genesets.keys()
	m, c = len(geneset_list), net_kernel.shape[0]
	subsample_mat = np.zeros((n*m, c))
	y_actual_mat = np.zeros((n*m, c))
	# Each block of length n rows is a sub-sampled binary vector of the corresponding gene set
	for i in range(m):
		geneset = geneset_list[i]
		# Get indices of gene set genes in kernel
		intersect = [gene for gene in genesets[geneset] if gene in net_kernel.index]
		index_dict = dict((gene, idx) for idx, gene in enumerate(net_kernel.index))
		intersect_idx = [index_dict[gene] for gene in intersect]
		# Generate n sub-samples
		for j in range(n):
			# Sub-sample gene set indices
			sample_size = int(round(genesets_p[geneset]*len(intersect)))
			sample_idx = random.sample(intersect_idx, sample_size)
			non_sample_idx = [idx for idx in intersect_idx if idx not in sample_idx]
			# Set sub-sampled list to 1
			row = (i*n)+j
			subsample_mat[row, sample_idx] = 1
			y_actual_mat[row, non_sample_idx] = 1
	if verbose:
		print 'Binary gene set sub-sample matrix constructed'
	# Propagate sub-samples
	prop_subsamples = np.dot(subsample_mat, net_kernel)
	if verbose:
		print 'Binary gene set sub-sample matrix propagated'
	# Construct parameter list to be passed
	AUPRC_Analysis_params = []
	for i in range(len(geneset_list)):
		AUPRCs = []
		for j in range(n):
			row = (i*n)+j
			prop_result = pd.DataFrame(np.array((subsample_mat[row], y_actual_mat[row], prop_subsamples[row])), 
				index=['Sub-Sample', 'Non-Sample', 'Prop Score'], 
				columns=net_kernel.columns).T.sort_values(by=['Sub-Sample', 'Prop Score', 'Non-Sample'],
					ascending=[False, False, False]).ix[int(sum(subsample_mat[row])):]['Non-Sample']
			intersect_non_sample_sorted = prop_result[prop_result==1].index
			P_totals = {node:float(prop_result.ix[:node].shape[0]) for node in intersect_non_sample_sorted}
			AUPRC_Analysis_params.append([geneset_list[i], intersect_non_sample_sorted, P_totals, verbose])
	# Determine parallel calculation status
	if cores == 1:
		# Calculate AUPRC values for all gene sets
		AUPRC_results = []
		for params_list in AUPRC_Analysis_params:
			AUPRC_results.append(calculate_large_network_AUPRC(params_list))
	else:
		# Initialize worker pool
		pool = Pool(cores)
		# Run the AUPRC analysis for each geneset
		AUPRC_results = pool.map(calculate_large_network_AUPRC, AUPRC_Analysis_params)
		# Close worker pool
		pool.close()		  
	# Construct AUPRC results
	geneset_AUPRCs = pd.DataFrame(AUPRC_results, columns=['Gene Set', 'AUPRCs']).set_index('Gene Set', drop=True)
	geneset_AUPRCs_merged = {geneset:geneset_AUPRCs.ix[geneset]['AUPRCs'].mean() for geneset in geneset_list}
	AUPRCs_table = pd.Series(geneset_AUPRCs_merged, name='AUPRC')
	return AUPRCs_table

# Wrapper to calculate AUPRCs of multiple node sets given network and node set files
def AUPRC_Analysis_single(network_file, genesets_file, shuffle=False, kernel_file=None, prop_constant=None, subsample_iter=30, cores=1, save_path=None, verbose=True):
	starttime = time.time()
	# Load network
	network = dit.load_network_file(network_file, verbose=verbose)
	# Shuffle network?
	if shuffle:
		network = shuffle_network(network, verbose=verbose)
	# Get network size
	net_size = len(network.edges())
	if verbose:
		print 'Network size:', net_size, 'Edges'
	# Calculate or load network propagation kernel
	if kernel_file is None:
		# Determine propagation constant
		if prop_constant is None:
			alpha = prop.calculate_alpha(network)
		else:
			alpha = prop_constant
		# Calculate network propagation kernel
		net_kernel = construct_prop_kernel(network, alpha=alpha, verbose=verbose)
	else:
		# Load network propagation kernel
		if kernel_file.endswith('.hdf'):
			net_kernel = pd.read_hdf(kernel_file)
		else:
			net_kernel = pd.read_csv(kernel_file)
	# Load node sets to recover
	genesets = dit.load_node_sets(genesets_file, verbose=verbose)
	# Calculate sub-sample rate for each node set given network
	genesets_p = calculate_p(network, genesets)
	# if network is small:
	if net_size < 250000:
		AUPRC_table = small_network_AUPRC_wrapper(net_kernel, genesets, genesets_p, n=subsample_iter, cores=cores, verbose=verbose)
	# if network is large:
	else:
		AUPRC_table = large_network_AUPRC_wrapper(net_kernel, genesets, genesets_p, n=subsample_iter, cores=cores, verbose=verbose)
	if verbose:
		print 'AUPRC values calculated', time.time()-starttime, 'seconds'
	# Save table
	if save_path is not None:
		AUPRC_table.to_csv(save_path)
	if verbose:
		print 'AUPRC table saved:', save_path		
	return AUPRC_table

# The function will take all files containing the filename marker given to shuff_net_AUPRCs_fn and construct a single null AUPRCs table from them (in wd)
# shuff_net_AUPRCs_fn is a generic filename marker (assumes all shuff_net_AUPRCs files have the same file name structure)
def get_null_AUPRCs_table(wd, shuff_net_AUPRCs_fn, geneset_list=None):
	shuff_net_AUPRCs = [pd.read_csv(wd+fn, index_col=0, header=-1) for fn in os.listdir(wd) if shuff_net_AUPRCs_fn in fn]
	shuff_net_AUPRCs = pd.concat(shuff_net_AUPRCs, axis=1)
	if geneset_list is None:
		return shuff_net_AUPRCs
	else:
		return shuff_net_AUPRCs.ix[geneset_list].dropna(axis=1)

# Calculate robust z-score metric for a network on given node sets given results directory of AUPRC calculations
# Requires the AUPRCs calculated for the actual network in a pandas Series
# Also requires the AUPRCs calculated for the same gene sets on the shuffled networks in a pandas DataFrame
def calculate_network_performance_score(actual_net_AUPRCs, shuff_net_AUPRCs, verbose=True, save_path=None):
	# Align data (only calculate for gene sets with full data on both actual networks and all shuffled networks)
	genesets = sorted(list(set(actual_net_AUPRCs.index).intersection(set(shuff_net_AUPRCs.index))), key=lambda s: s.lower())
	actual_net_AUPRCs = actual_net_AUPRCs.ix[genesets]
	shuff_net_AUPRCs = shuff_net_AUPRCs.ix[genesets]
	# Compute robust z-score for composite network performances
	k = 1/stats.norm.ppf(0.75)	# Mean absolute deviation scaling factor to make median absolute deviation behave similarly to the standard deviation of a normal distribution
	AUPRC_null_median = shuff_net_AUPRCs.median(axis=1)
	AUPRC_null_MAD = abs(shuff_net_AUPRCs.subtract(AUPRC_null_median, axis=0)).median(axis=1)
	AUPRC_null_MAD_scaled = k*AUPRC_null_MAD
	AUPRC_ZNorm = (actual_net_AUPRCs[1] - AUPRC_null_median).divide(AUPRC_null_MAD_scaled)
	if save_path is not None:
		AUPRC_ZNorm.to_csv(save_path)
	if verbose:
		print 'AUPRC values z-normalized'				
	return AUPRC_ZNorm

# Calculate relative gain of actual network AUPRC over median random network AUPRC performance for each gene set
# Requires the AUPRCs calculated for the actual network in a pandas Series
# Also requires the AUPRCs calculated for the same gene sets on the shuffled networks in a pandas DataFrame
def calculate_network_performance_gain(actual_net_AUPRCs, shuff_net_AUPRCs, verbose=True, save_path=None):
	# Align data (only calculate for gene sets with full data on both actual networks and all shuffled networks)
	genesets = sorted(list(set(actual_net_AUPRCs.index).intersection(set(shuff_net_AUPRCs.index))), key=lambda s: s.lower())
	actual_net_AUPRCs = actual_net_AUPRCs.ix[genesets]
	shuff_net_AUPRCs = shuff_net_AUPRCs.ix[genesets]	
	# Compute relative gain
	AUPRC_null_median = shuff_net_AUPRCs.median(axis=1)
	AUPRC_gain = (actual_net_AUPRCs[1] - AUPRC_null_median).divide(AUPRC_null_median)
	if save_path is not None:
		AUPRC_gain.to_csv(save_path)
	if verbose:
		print 'AUPRC relative performance gain calculated'
	return AUPRC_gain

