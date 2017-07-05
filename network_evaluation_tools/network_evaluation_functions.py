#####################################################################
# ---------- Node Set-Based Network Evaluation Functions ---------- #
#####################################################################
import networkx as nx
import time
import numpy as np
import pandas as pd
import random
import scipy.stats as stats
import sklearn.metrics as metrics
import data_import_tools as dit
import network_propagation as prop
from multiprocessing import Pool
import os
import pickle as p

# Shuffle network in degree-preserving manner
# Input: network - networkx formatted network
def shuffle_network(network, verbose=False):
	# Shuffle Network
	shuff_time = time.time()
	edge_len=len(network.edges())
	shuff_net=network.copy()
	try:
		nx.double_edge_swap(shuff_net, nswap=edge_len, max_tries=edge_len*10)
	except:
		if verbose:
			print 'Note: Maximum number of swap attempts ('+repr(edge_len*10)+') exceeded before desired swaps achieved ('+repr(edge_len)+').'
	if verbose:
		# Evaluate Network Similarity
		shared_edges = len(set(network.edges()).intersection(set(shuff_net.edges())))
		print 'Network shuffled:', time.time()-shuff_time, 'seconds. Edge similarity:', shared_edges/float(edge_len)
	return shuff_net

# Calculate optimal sub-sampling proportion for test/train
def calculate_p(network, nodesets, m=-0.18887257, b=0.64897403):
	network_nodes = [str(gene) for gene in network.nodes()]
	nodesets_p = {}
	for nodeset in nodesets:
		nodesets_coverage = len([node for node in nodeset if node in network_nodes])
		nodesets_p[nodeset] = round(m*nodesets_coverage+b, 3)
	return nodesets_p

# Construct influence matrix of each network node propagated across network to use as kernel in AUPRC analysis
def construct_prop_kernel(network, alpha=None, m=-0.17190024, b=0.7674828, verbose=False):
	network_Fo = pd.DataFrame(data=np.identity(len(network.nodes())), index=network.nodes(), columns=network.nodes())
	if alpha is None:
		alpha_val = prop.calculate_alpha(network, m=m, b=b)
	else:
		alpha_val = alpha
	network_Fn = prop.closed_form_network_propagation(network, network_Fo, alpha_val, verbose=verbose)
	if verbose:
		print 'Propagated network kernel constructed'
	return network_Fn

# Analyze AUPRC of node set recovery for given node set (parameter setup written for running in serial)
def calculate_AUPRC_serial(prop_geno, p, n, node_set, verbose=False):
	runtime = time.time()
	intersect = [nodes for nodes in node_set if nodes in prop_geno.index]
	AUPRCs = []
	sample_size = int(round(p*len(intersect)))
	for i in range(n):																					  	# Number of times to run the sampling
		sample = random.sample(intersect, sample_size)														 	# get node set sample
		intersect_non_sample = [node for node in intersect if node not in sample]							   	# nodes in intersect not in sample
		prop_geno_non_sample = list(prop_geno.index[~prop_geno.index.isin(sample)])							 	# nodes in network not in sample
		prop_geno_sample_sum = prop_geno.ix[sample][prop_geno_non_sample].sum().sort_values(ascending=False)	# summed prop value for all nodes
		y_actual = pd.Series(0, index=prop_geno_sample_sum.index, dtype=int)									# nodes sorted by mean prop value
		y_actual.ix[intersect_non_sample]+=1																	# which nodes in sorted list are in intersect_non_sample
		intersect_non_sample_sorted = y_actual[y_actual==1].index											   	# intersect_non_sample sorted
		precision, recall = [1], [0]																		  	# initialize precision and recall curves
		for node in intersect_non_sample_sorted:															# Slide down sorted nodes by summed prop value by nodes that are in intersect_non_sample
			TP, FN = sum(y_actual.ix[:node]), sum(y_actual.ix[node:])-1										   	# Calculate true positives and false negatives found at this point in list
			precision.append(TP/float(y_actual.ix[:node].shape[0]))											 	# Calculate precision ( TP / TP+FP ) and add point to curve
			recall.append(TP/float(TP+FN))																	  	# Calculate recall ( TP / TP+FN ) and add point to curve
		AUPRCs.append(metrics.auc(recall, precision))													   		# Calculate Area Under Precision-Recall Curve (AUPRC)
	if verbose:
		print 'AUPRC Analysis for given node set', '('+repr(len(intersect))+' nodes in network) complete:', round(time.time()-runtime, 2), 'seconds.'
	return np.mean(AUPRCs)

# Analyze AUPRC of node set recovery for given node set (parameter setup written for running in serial)
def calculate_AUPRC_parallel(node_set_params):
	node_set_name, node_set, p, n, verbose = node_set_params[0], node_set_params[1], node_set_params[2], node_set_params[3], node_set_params[4]
	runtime = time.time()
	intersect = [nodes for nodes in node_set if nodes in prop_geno.index]
	AUPRCs = []
	sample_size = int(round(p*len(intersect)))
	for i in range(n):																					  	# Number of times to run the sampling
		sample = random.sample(intersect, sample_size)														 	# get node set sample
		intersect_non_sample = [node for node in intersect if node not in sample]							   	# nodes in intersect not in sample
		prop_geno_non_sample = list(prop_geno.index[~prop_geno.index.isin(sample)])							 	# nodes in network not in sample
		prop_geno_sample_sum = prop_geno.ix[sample][prop_geno_non_sample].sum().sort_values(ascending=False)	# summed prop value for all nodes
		y_actual = pd.Series(0, index=prop_geno_sample_sum.index, dtype=int)									# nodes sorted by mean prop value
		y_actual.ix[intersect_non_sample]+=1																	# which nodes in sorted list are in intersect_non_sample
		intersect_non_sample_sorted = y_actual[y_actual==1].index											   	# intersect_non_sample sorted
		precision, recall = [1], [0]																		  	# initialize precision and recall curves
		for node in intersect_non_sample_sorted:															# Slide down sorted nodes by summed prop value by nodes that are in intersect_non_sample
			TP, FN = sum(y_actual.ix[:node]), sum(y_actual.ix[node:])-1										   	# Calculate true positives and false negatives found at this point in list
			precision.append(TP/float(y_actual.ix[:node].shape[0]))											 	# Calculate precision ( TP / TP+FP ) and add point to curve
			recall.append(TP/float(TP+FN))																	  	# Calculate recall ( TP / TP+FN ) and add point to curve
		AUPRCs.append(metrics.auc(recall, precision))													   		# Calculate Area Under Precision-Recall Curve (AUPRC)
	if verbose:
		print 'AUPRC Analysis for given node set', '('+repr(len(intersect))+' nodes in network) complete:', round(time.time()-runtime, 2), 'seconds.'
	return [node_set_name, np.mean(AUPRCs)]

# Initializer function for defining global variables for each thread if running AUPRC Analysis in parallel
def parallel_analysis_initializer(global_prop_net):
	global prop_geno
	prop_geno = global_prop_net

# Wapper for conducting AUPRC Analysis for input node set file and network (has parallel option)
def AUPRC_Analysis(network_file, node_set_file, sample_p, sub_sample_iterations, 
	alpha=None, sample_m=-0.18887257, sample_b=0.64897403, prop_m=-0.02935302, prop_b=0.74842057, 
	net_delim='\t', set_delim='\t', cores=1, verbose=False, save_path=None):
	starttime=time.time()
	# Load network
	network = dit.load_network_file(network_file, delimiter=net_delim, verbose=verbose)
	# Load node set
	node_sets = dit.load_node_sets(node_set_file, delimiter=set_delim, verbose=verbose)
	# Calculate p for each node set
	node_sets_p = calculate_p(network, nodesets, m=sample_m, b=sample_b)
	# Calculate network influence matrix
	prop_net = construct_prop_kernel(network, alpha=alpha, m=prop_m, b=prop_b, verbose=verbose)
	# Calculate AUPRC values for each node set
	if cores == 1:
		# Calculate AUPRC values for node sets one at a time
		node_set_AUPRCs = {node_set:calculate_AUPRC_serial(prop_net, node_sets_p[node_set], sub_sample_iterations, node_sets[node_set], verbose=False) for node_set in node_sets}
	else:
		# Initialize multiple threads for AUPRC analysis of multiple node sets
		initializer_args = [prop_net]
		pool = Pool(cores, parallel_analysis_initializer, initializer_args)
		# Construct parameter list to be passed
		AUPRC_Analysis_params = [[node_set, node_sets[node_set], node_sets_p[node_set], sub_sample_iterations, verbose] for node_set in node_sets]
		# Run the AUPRC analysis for each geneset
		AUPRC_results = pool.map(calculate_AUPRC_parallel, AUPRC_Analysis_params)
		# Construct AUPRC results dictionary
		node_set_AUPRCs = {result[0]:result[1] for result in AUPRC_results}
	AUPRCs_table = pd.DataFrame(pd.Series(node_set_AUPRCs, name='AUPRC'))
	if save_path is None:
		if verbose:
			print 'Network AUPRC Analysis complete:', round(time.time()-starttime, 2), 'seconds'			
		return AUPRCs_table
	else:
		AUPRCs_table.to_csv(save_path)
		if verbose:
			print 'Network AUPRC Analysis complete:', round(time.time()-starttime, 2), 'seconds'			
		return AUPRCs_table

# Wrapper for shuffling input network and performing AUPRC analysis on each shuffled network and then compile results
def null_AUPRC_Analysis(network_file, node_set_file, sample_p, sub_sample_iterations, null_iterations, 
	alpha=None, sample_m=-0.18887257, sample_b=0.64897403, prop_m=-0.02935302, prop_b=0.74842057, 
	net_delim='\t', set_delim='\t', cores=1, verbose=False, save_path=None):
	starttime=time.time()
	# Load network
	network = dit.load_network_file(network_file, delimiter=net_delim, verbose=verbose)
	# Load node set
	node_sets = dit.load_node_sets(node_set_file, delimiter=set_delim, verbose=verbose)
	# Calculate p for each node set
	node_sets_p = calculate_p(network, nodesets, m=sample_m, b=sample_b)
	# Analyze shuffled networks
	null_AUPRCs = []
	for i in range(null_iterations):
		shuff_net = shuffle_network(network)
		prop_shuff_net = construct_prop_kernel(shuff_net, alpha=alpha, m=m, b=b)
		# Calculate AUPRC values for each node set
		if cores == 1:
			# Calculate AUPRC values for node sets one at a time
			node_set_AUPRCs = {node_set:calculate_AUPRC_serial(prop_shuff_net, node_sets_p[node_set], sub_sample_iterations, node_sets[node_set], verbose=False) for node_set in node_sets}
		else:
			# Initialize multiple threads for AUPRC analysis of multiple node sets
			initializer_args = [prop_shuff_net]
			pool = Pool(cores, parallel_analysis_initializer, initializer_args)
			# Construct parameter list to be passed
			AUPRC_Analysis_params = [[node_set, node_sets[node_set], node_sets_p[node_set], sub_sample_iterations, False] for node_set in node_sets]
			# Run the AUPRC analysis for each geneset
			AUPRC_results = pool.map(calculate_AUPRC_parallel, AUPRC_Analysis_params)
			pool.close()
			# Construct AUPRC results dictionary
			node_set_AUPRCs = {result[0]:result[1] for result in AUPRC_results}
		null_AUPRCs.append(pd.Series(node_set_AUPRCs, name='null AUPRC '+repr(i+1)))
		if verbose: # All of the verbosity for each shuffled network is turned off to prevent cluttering of the log
			print 'Shuffled Network', repr(i+1), 'AUPRC Analysis done'
	null_AUPRCs_table = pd.concat(null_AUPRCs, axis=1)
	if save_path is None:
		if verbose:
			print 'Null AUPRC Analysis complete:', round(time.time()-starttime, 2), 'seconds'
		return null_AUPRCs_table
	else:
		null_AUPRCs_table.to_csv(save_path)
		if verbose:
			print 'Null AUPRC Analysis complete:', round(time.time()-starttime, 2), 'seconds'		
		return null_AUPRCs_table

# Calculate robust z-score metric for a network on given node sets given results of AUPRC_Analysis_wrapper and null_AUPRC_Analysis_wrapper
def AUPRC_Analysis_with_ZNorm(actual_net_AUPRCs_path, shuff_net_AUPRCs_path, verbose=False, save_path=None):
	# Read input data files and concat together:
	actual_net_AUPRCs = pd.read_csv(actual_net_AUPRCs_path, index_col=0)
	shuff_net_AUPRCs = pd.read_csv(shuff_net_AUPRCs_path, index_col=0)
	shuff_net_AUPRCs = shuff_net_AUPRCs.ix[actual_net_AUPRCs.index]
	# Compute robust z-score for composite network performances
	k = 1/stats.norm.ppf(0.75)	# Mean absolute deviation scaling factor to make median absolute deviation behave similarly to the standard deviation of a normal distribution
	AUPRC_null_median = shuff_net_AUPRCs.median(axis=1)
	AUPRC_null_MAD = abs(shuff_net_AUPRCs.subtract(AUPRC_null_median, axis=0)).median(axis=1)
	AUPRC_null_MAD_scaled = k*AUPRC_null_MAD
	AUPRC_ZNorm = (actual_net_AUPRCs['AUPRC'] - AUPRC_null_median).divide(AUPRC_null_MAD_scaled)
	if save_path is None:
		if verbose:
			print 'AUPRC values z-normalized'		
		return AUPRC_ZNorm
	else:
		AUPRC_ZNorm.to_csv(save_path)
		if verbose:
			print 'AUPRC values z-normalized'				
		return AUPRC_ZNorm

