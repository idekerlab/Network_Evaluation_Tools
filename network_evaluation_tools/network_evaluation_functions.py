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
import network_propagation as prop

# Construct dictionary of node sets from input file
def load_node_sets(node_set_file):
	f = open(node_set_file)
	node_set_lines = f.read().splitlines()
	node_set_lines_split = [line.split('\t') for line in node_set_lines]
	f.close()
	node_sets = {node_set[0]:set(node_set[1:]) for node_set in node_set_lines_split}
	return node_sets

# Shuffle network in degree-preserving manner
# Input: network - networkx formatted network
def shuffle_network(network):
	# Shuffle Network
	starttime = time.time()
	edge_len=len(network.edges())
	shuff_net=network.copy()
	nx.double_edge_swap(shuff_net, nswap=edge_len, max_tries=edge_len*10)
	# Evaluate Network Similarity
	shared_edges = len(set(network.edges()).intersection(set(shuff_net.edges())))
	print 'Network shuffled:', time.time()-shuff_time, 'seconds. Edge similarity:', shared_edges/float(len(network.edges()))
	return shuff_net

# Construct influence matrix of each network node propagated across network to use as kernel in AUPRC analysis
def construct_prop_kernel(network):
	network_Fo = pd.DataFrame(data=np.identity(len(network.nodes())), index=network.nodes(), columns=network.nodes())
	network_Fn = prop.closed_form_network_propagation(network, network_Fo)
	return network_Fn

# Analyze AUPRC of node set recovery for given node set (parameter setup written for running in serial)
def calculate_AUPRC_serial(prop_geno, p, n, node_set):
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
		precision, recall = [], []																			  	# initialize precision and recall curves
		for node in intersect_non_sample_sorted:															# Slide down sorted nodes by summed prop value by nodes that are in intersect_non_sample
			TP, FN = sum(y_actual.ix[:node]), sum(y_actual.ix[node:])										   	# Calculate true positives and false negatives found at this point in list
			precision.append(TP/float(y_actual.ix[:node].shape[0]))											 	# Calculate precision ( TP / TP+FP ) and add point to curve
			recall.append(TP/float(TP+FN))																	  	# Calculate recall ( TP / TP+FN ) and add point to curve
		precision = [1]+precision+[0]
		recall=[0]+recall+[1]
		AUPRCs.append(metrics.auc(recall, precision))													   		# Calculate Area Under Precision-Recall Curve (AUPRC)
	print 'AUPRC Analysis for given node set', '('+repr(len(intersect))+' nodes in network) complete:', round(time.time()-runtime, 2), 'seconds.'
	return np.mean(AUPRCs)

# Analyze AUPRC of node set recovery for given node set (parameter setup written for running in serial)
def calculate_AUPRC_parallel(node_set_params):
	node_set_name, node_set, p, n = node_set_params[0], node_set_params[1], node_set_params[2], node_set_params[3]
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
		precision, recall = [], []																			  	# initialize precision and recall curves
		for node in intersect_non_sample_sorted:															# Slide down sorted nodes by summed prop value by nodes that are in intersect_non_sample
			TP, FN = sum(y_actual.ix[:node]), sum(y_actual.ix[node:])										   	# Calculate true positives and false negatives found at this point in list
			precision.append(TP/float(y_actual.ix[:node].shape[0]))											 	# Calculate precision ( TP / TP+FP ) and add point to curve
			recall.append(TP/float(TP+FN))																	  	# Calculate recall ( TP / TP+FN ) and add point to curve
		precision = [1]+precision+[0]
		recall=[0]+recall+[1]
		AUPRCs.append(metrics.auc(recall, precision))													   		# Calculate Area Under Precision-Recall Curve (AUPRC)
	print 'AUPRC Analysis for given node set', '('+repr(len(intersect))+' nodes in network) complete:', round(time.time()-runtime, 2), 'seconds.'
	return [node_set_name, np.mean(AUPRCs)]

# Initializer function for defining global variables for each thread if running AUPRC Analysis in parallel
def AUPRC_Analysis_initializer(global_prop_net):
	global prop_geno
	prop_geno = global_prop_net

# Wapper for conducting AUPRC Analysis for input node set file and network (has parallel option)
def AUPRC_Analysis(network_file, node_set_file, sample_p, AUPRC_iterations, cores=1, save_results=False, outdir=None):
	# Load network
	network = prop.load_network(network_file, delimiter='\t')
	# Load node set
	node_sets = load_node_sets(node_set_file)
	# Calculate network influence matrix
	prop_net = construct_prop_kernel(network)
	# Calculate AUPRC values for each node set
	if cores == 1:
		# Calculate AUPRC values for node sets one at a time
		node_set_AUPRCs = {node_set:calculate_AUPRC_serial(prop_net, sample_p, AUPRC_iterations, node_sets[node_set]) for node_set in node_sets}
	else:
		# Initialize multiple threads for AUPRC analysis of multiple node sets
		initializer_args = [prop_net]
		pool = Pool(cores, AUPRC_Analysis_initializer, initializer_args)
		# Construct parameter list to be passed
		AUPRC_Analysis_params = [[node_sets, node_sets[node_set], sample_p, AUPRC_iterations] for node_set in node_sets]
		# Run the AUPRC analysis for each geneset
		AUPRC_results = pool.map(calculate_AUPRC_serial, AUPRC_Analysis_params)
		# Construct AUPRC results dictionary
		node_set_AUPRCs = {result[0]:result[1] for result in AUPRC_results}
	if save_results == False:
		return node_set_AUPRCs
	else:
		pd.Series(node_set_AUPRCs, name='AUPRC').to_csv(outdir+'AUPRC_results.csv')
		return node_set_AUPRCs

# Wrapper for shuffling input network and performing AUPRC analysis on each shuffled network and then compile results
def null_AUPRC_Analysis_wrapper(network_file, node_set_file, sample_p, AUPRC_iterations, null_iterations, cores=1, save_results=False, outdir=None):
	# Load network
	network = prop.load_network_file(network_file, delimiter='\t')
	# Load node set
	node_sets = load_node_sets(node_set_file)
	# Analyze shuffled networks
	null_AUPRCs = []
	for i in range(len(null_iterations)):
		shuff_net = shuffle_network(network)
		prop_shuff_net = construct_prop_kernel(shuff_net)
		# Calculate AUPRC values for each node set
		if cores == 1:
			# Calculate AUPRC values for node sets one at a time
			node_set_AUPRCs = {node_set:calculate_AUPRC_serial(prop_shuff_net, sample_p, AUPRC_iterations, node_sets[node_set]) for node_set in node_sets}
		else:
			# Initialize multiple threads for AUPRC analysis of multiple node sets
			initializer_args = [prop_shuff_net]
			pool = Pool(cores, AUPRC_Analysis_initializer, initializer_args)
			# Construct parameter list to be passed
			AUPRC_Analysis_params = [[node_sets, node_sets[node_set], sample_p, AUPRC_iterations] for node_set in node_sets]
			# Run the AUPRC analysis for each geneset
			AUPRC_results = pool.map(calculate_AUPRC_serial, AUPRC_Analysis_params)
			# Construct AUPRC results dictionary
			node_set_AUPRCs = {result[0]:result[1] for result in AUPRC_results}
		null_AUPRCs.append(pd.Series(node_set_AUPRCs, name='null AUPRC '+repr(i)))
	null_AUPRCs_table = pd.cocnat(null_AUPRCs, axis=1)

	if save_results == False:
		return null_AUPRCs_table
	else:
		null_AUPRCs_table.to_csv(outdir+'null_AUPRC_results.csv')
		return null_AUPRCs_table

# Calculate robust z-score metric for a network on given node sets given results of AUPRC_Analysis_wrapper and null_AUPRC_Analysis_wrapper
def AUPRC_Analysis_with_ZNorm(actual_net_AUPRCs, shuff_net_AUPRCs, save_results=False, outdir=None):
	# Read input data files and concat together:
	shuff_net_AUPRCs = pd.read_csv(shuff_net_AUPRCs, index_col=0)
	actual_net_AUPRCs = pd.read_csv(shuff_net_AUPRCs, index_col=0)

	k = 1/stats.norm.ppf(0.75)	# Mean absolute deviation scaling factor to make median absolute deviation behave similarly to the standard deviation of a normal distribution
	# Compute robust z-score for composite network performances
	AUPRC_null_median = shuff_net_AUPRCs.median(axis=1)
	AUPRC_null_MAD = abs(actual_net_AUPRCs.ix[AUPRC_null_median.index].subtract(AUPRC_null_median, axis=0)).median(axis=1)
	AUPRC_null_MAD_scaled = k*AUPRC_null_MAD
	AUPRC_ZNorm = (shuff_net_AUPRCs.ix[AUPRC_null_median.index] - AUPRC_null_median).divide(AUPRC_null_MAD_scaled)
	if save_results == False:
		return AUPRC_ZNorm
	else:
		AUPRC_ZNorm.to_csv(outdir+'AUPRC_results_ZNorm.csv')
		return AUPRC_ZNorm























