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
from multiprocessing import Pool
import os
import pickle as p

# Construct dictionary of node sets from input file
def load_node_sets(node_set_file, delimiter='\t'):
	f = open(node_set_file)
	node_set_lines = f.read().splitlines()
	node_set_lines_split = [line.split(delimiter) for line in node_set_lines]
	f.close()
	node_sets = {node_set[0]:set(node_set[1:]) for node_set in node_set_lines_split}
	return node_sets

# Shuffle network in degree-preserving manner
# Input: network - networkx formatted network
def shuffle_network(network):
	# Shuffle Network
	shuff_time = time.time()
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
		precision, recall = [1], [0]																		  	# initialize precision and recall curves
		for node in intersect_non_sample_sorted:															# Slide down sorted nodes by summed prop value by nodes that are in intersect_non_sample
			TP, FN = sum(y_actual.ix[:node]), sum(y_actual.ix[node:])-1										   	# Calculate true positives and false negatives found at this point in list
			precision.append(TP/float(y_actual.ix[:node].shape[0]))											 	# Calculate precision ( TP / TP+FP ) and add point to curve
			recall.append(TP/float(TP+FN))																	  	# Calculate recall ( TP / TP+FN ) and add point to curve
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
		precision, recall = [1], [0]																		  	# initialize precision and recall curves
		for node in intersect_non_sample_sorted:															# Slide down sorted nodes by summed prop value by nodes that are in intersect_non_sample
			TP, FN = sum(y_actual.ix[:node]), sum(y_actual.ix[node:])-1										   	# Calculate true positives and false negatives found at this point in list
			precision.append(TP/float(y_actual.ix[:node].shape[0]))											 	# Calculate precision ( TP / TP+FP ) and add point to curve
			recall.append(TP/float(TP+FN))																	  	# Calculate recall ( TP / TP+FN ) and add point to curve
		AUPRCs.append(metrics.auc(recall, precision))													   		# Calculate Area Under Precision-Recall Curve (AUPRC)
	print 'AUPRC Analysis for given node set', '('+repr(len(intersect))+' nodes in network) complete:', round(time.time()-runtime, 2), 'seconds.'
	return [node_set_name, np.mean(AUPRCs)]

# Initializer function for defining global variables for each thread if running AUPRC Analysis in parallel
def parallel_analysis_initializer(global_prop_net):
	global prop_geno
	prop_geno = global_prop_net

# Wapper for conducting AUPRC Analysis for input node set file and network (has parallel option)
def AUPRC_Analysis(network_file, node_set_file, sample_p, sub_sample_iterations, cores=1, save_results=False, outdir=None):
	# Load network
	network = prop.load_network_file(network_file, delimiter='\t')
	# Load node set
	node_sets = load_node_sets(node_set_file)
	# Calculate network influence matrix
	prop_net = construct_prop_kernel(network)
	# Calculate AUPRC values for each node set
	if cores == 1:
		# Calculate AUPRC values for node sets one at a time
		node_set_AUPRCs = {node_set:calculate_AUPRC_serial(prop_net, sample_p, sub_sample_iterations, node_sets[node_set]) for node_set in node_sets}
	else:
		# Initialize multiple threads for AUPRC analysis of multiple node sets
		initializer_args = [prop_net]
		pool = Pool(cores, AUPRC_Analysis_initializer, initializer_args)
		# Construct parameter list to be passed
		AUPRC_Analysis_params = [[node_set, node_sets[node_set], sample_p, sub_sample_iterations] for node_set in node_sets]
		# Run the AUPRC analysis for each geneset
		AUPRC_results = pool.map(calculate_AUPRC_parallel, AUPRC_Analysis_params)
		# Construct AUPRC results dictionary
		node_set_AUPRCs = {result[0]:result[1] for result in AUPRC_results}
	AUPRCs_table = pd.DataFrame(pd.Series(node_set_AUPRCs, name='AUPRC'))
	if save_results == False:
		return AUPRCs_table
	else:
		AUPRCs_table.to_csv(outdir+'AUPRC_results.csv')
		return AUPRCs_table

# Wrapper for shuffling input network and performing AUPRC analysis on each shuffled network and then compile results
def null_AUPRC_Analysis(network_file, node_set_file, sample_p, sub_sample_iterations, null_iterations, cores=1, save_results=False, outdir=None):
	# Load network
	network = prop.load_network_file(network_file, delimiter='\t')
	# Load node set
	node_sets = load_node_sets(node_set_file)
	# Analyze shuffled networks
	null_AUPRCs = []
	for i in range(null_iterations):
		shuff_net = shuffle_network(network)
		prop_shuff_net = construct_prop_kernel(shuff_net)
		# Calculate AUPRC values for each node set
		if cores == 1:
			# Calculate AUPRC values for node sets one at a time
			node_set_AUPRCs = {node_set:calculate_AUPRC_serial(prop_shuff_net, sample_p, sub_sample_iterations, node_sets[node_set]) for node_set in node_sets}
		else:
			# Initialize multiple threads for AUPRC analysis of multiple node sets
			initializer_args = [prop_shuff_net]
			pool = Pool(cores, parallel_analysis_initializer, initializer_args)
			# Construct parameter list to be passed
			AUPRC_Analysis_params = [[node_set, node_sets[node_set], sample_p, sub_sample_iterations] for node_set in node_sets]
			# Run the AUPRC analysis for each geneset
			AUPRC_results = pool.map(calculate_AUPRC_parallel, AUPRC_Analysis_params)
			# Construct AUPRC results dictionary
			node_set_AUPRCs = {result[0]:result[1] for result in AUPRC_results}
		null_AUPRCs.append(pd.Series(node_set_AUPRCs, name='null AUPRC '+repr(i+1)))
	null_AUPRCs_table = pd.concat(null_AUPRCs, axis=1)

	if save_results == False:
		return null_AUPRCs_table
	else:
		null_AUPRCs_table.to_csv(outdir+'null_AUPRC_results.csv')
		return null_AUPRCs_table

# Calculate robust z-score metric for a network on given node sets given results of AUPRC_Analysis_wrapper and null_AUPRC_Analysis_wrapper
def AUPRC_Analysis_with_ZNorm(actual_net_AUPRCs_path, shuff_net_AUPRCs_path, save_results=False, outdir=None):
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
	if save_results == False:
		return AUPRC_ZNorm
	else:
		AUPRC_ZNorm.to_csv(outdir+'AUPRC_results_ZNorm.csv')
		return AUPRC_ZNorm

################################################################################
# ---------- Additional Node Set-Based Network Evaluation Functions ---------- #
################################################################################

# Calculate confusion matrix (true positive, false negatives, false positives, true negatives) of node set recovery for given node set
# The confusion matrix for every position on every AUPRC curve is returned/stored
def calculate_confusion_matrix_serial(prop_geno, p, n, node_set_name, node_set):
	runtime = time.time()
	intersect = [nodes for nodes in node_set if nodes in prop_geno.index]
	confusion_matrices = {}
	sample_size = int(round(p*len(intersect)))
	for i in range(n):																					  	# Number of times to run the sampling
		sample = random.sample(intersect, sample_size)														 	# get node set sample
		intersect_non_sample = [node for node in intersect if node not in sample]							   	# nodes in intersect not in sample
		prop_geno_non_sample = list(prop_geno.index[~prop_geno.index.isin(sample)])							 	# nodes in network not in sample
		prop_geno_sample_sum = prop_geno.ix[sample][prop_geno_non_sample].sum().sort_values(ascending=False)	# summed prop value for all nodes
		y_actual = pd.Series(0, index=prop_geno_sample_sum.index, dtype=int)									# nodes sorted by mean prop value
		y_actual.ix[intersect_non_sample]+=1																	# which nodes in sorted list are in intersect_non_sample
		intersect_non_sample_sorted = y_actual[y_actual==1].index											   	# intersect_non_sample sorted
		confusion_matrix = {'TP':[], 'FN':[], 'FP':[], 'TN':[]}													# initialize true positive, false negative, false positive, true negative lists
		for node in intersect_non_sample_sorted:															# Slide down sorted nodes by summed prop value by nodes that are in intersect_non_sample
			TP, FN = sum(y_actual.ix[:node]), sum(y_actual.ix[node:])-1										   	# Calculate true positives and false negatives found at this point in list
			FP, TN = len(y_actual.ix[:node])-TP, len(y_actual.ix[node:])-1-FN									# Calculate false positives and true negatives found at this point in list
			confusion_matrix['TP'].append(TP)
			confusion_matrix['FN'].append(FN)
			confusion_matrix['FP'].append(FP)
			confusion_matrix['TN'].append(TN)
		confusion_matrices[i]=confusion_matrix
	print 'Confusion matrices calculated for node set', node_set_name, 'complete.', repr(len(intersect))+' nodes in network,', round(time.time()-runtime, 2), 'seconds.'
	return confusion_matrices

# Calculate confusion matrix (true positive, false negatives, false positives, true negatives) of node set recovery for given node set 
# The parameter setup is written for running in serial, only difference is that the name of the node set also must be passed, and prop_geno will be set as a global variable
# The confusion matrix for every position on every AUPRC curve is returned/stored
def calculate_confusion_matrix_parallel(node_set_params):
	node_set_name, node_set, p, n = node_set_params[0], node_set_params[1], node_set_params[2], node_set_params[3]
	runtime = time.time()
	intersect = [nodes for nodes in node_set if nodes in prop_geno.index]
	confusion_matrices = {}
	sample_size = int(round(p*len(intersect)))
	for i in range(n):																					  	# Number of times to run the sampling
		sample = random.sample(intersect, sample_size)														 	# get node set sample
		intersect_non_sample = [node for node in intersect if node not in sample]							   	# nodes in intersect not in sample
		prop_geno_non_sample = list(prop_geno.index[~prop_geno.index.isin(sample)])							 	# nodes in network not in sample
		prop_geno_sample_sum = prop_geno.ix[sample][prop_geno_non_sample].sum().sort_values(ascending=False)	# summed prop value for all nodes
		y_actual = pd.Series(0, index=prop_geno_sample_sum.index, dtype=int)									# nodes sorted by mean prop value
		y_actual.ix[intersect_non_sample]+=1																	# which nodes in sorted list are in intersect_non_sample
		intersect_non_sample_sorted = y_actual[y_actual==1].index											   	# intersect_non_sample sorted
		confusion_matrix = {'TP':[], 'FN':[], 'FP':[], 'TN':[]}													# initialize true positive, false negative, false positive, true negative lists
		for node in intersect_non_sample_sorted:															# Slide down sorted nodes by summed prop value by nodes that are in intersect_non_sample
			TP, FN = sum(y_actual.ix[:node]), sum(y_actual.ix[node:])-1										   	# Calculate true positives and false negatives found at this point in list
			FP, TN = len(y_actual.ix[:node])-TP, len(y_actual.ix[node:])-1-FN									# Calculate false positives and true negatives found at this point in list
			confusion_matrix['TP'].append(TP)
			confusion_matrix['FN'].append(FN)
			confusion_matrix['FP'].append(FP)
			confusion_matrix['TN'].append(TN)
		confusion_matrices[i]=confusion_matrix
	print 'Confusion matrices calculated for node set', node_set_name, 'complete.', repr(len(intersect))+' nodes in network,', round(time.time()-runtime, 2), 'seconds.'
	return [node_set_name, confusion_matrices]

# Wapper for calculating the confusion matrices for input node set file and network (has parallel option)
# Not run for null network shuffles
def confusion_matrix_construction_wrapper(network_file, node_set_file, sample_p, sub_sample_iterations, cores=1, save_results=False, outdir=None):
	# Load network
	network = prop.load_network_file(network_file, delimiter='\t')
	# Load node set
	node_sets = load_node_sets(node_set_file)
	# Calculate network influence matrix
	prop_net = construct_prop_kernel(network)
	# Calculate confusion matrix values for each node set
	if cores == 1:
		# Calculate confusion matrix values for node sets one at a time
		node_set_conf_mat = {node_set:calculate_confusion_matrix_serial(prop_net, sample_p, sub_sample_iterations, node_set, node_sets[node_set]) for node_set in node_sets}
	else:
		# Initialize multiple threads for confusion matrix analysis of multiple node sets
		initializer_args = [prop_net]
		pool = Pool(cores, parallel_analysis_initializer, initializer_args)
		# Construct parameter list to be passed
		conf_mat_Analysis_params = [[node_set, node_sets[node_set], sample_p, sub_sample_iterations] for node_set in node_sets]
		# Run the confusion matrix analysis for each geneset
		conf_mat_results = pool.map(calculate_confusion_matrix_parallel, conf_mat_Analysis_params)
		# Construct confusion matrix results dictionary
		node_set_conf_mat = {result[0]:result[1] for result in conf_mat_results}
	if save_results == False:
		return node_set_conf_mat
	else:
		p.dump(node_set_conf_mat, open(outdir+'confusion_matrix_results.p', 'wb'))
		return node_set_conf_mat

# Use confusion matrix results to calculate odds ratio, risk ratio, accuracy or precision at a given recall threshold
def confusion_matrix_analysis(confusion_matrix_input, calculation, recall_threshold=0.9, save_results=False, outdir=None):
	runtime = time.time()
	# Load confusion matrix data
	if type(confusion_matrix_input)!=dict:
		confusion_matrix = p.load(open(confusion_matrix_input, 'rb'))
	else:
		confusion_matrix = confusion_matrix_input

	# Calculate average and variance of specified calculation
	cohort_calculated_values_mean, cohort_calculated_values_var = {}, {}
	# For each cohort tested
	for cohort in confusion_matrix:
		print cohort
		n = len(confusion_matrix[cohort])
		calculation_values = []
		# For all sub-sample iterations
		for i in range(n):
			# Find where recall >= recall threshold
			for j in range(len(confusion_matrix[cohort][i]['TP'])):
				TP = confusion_matrix[cohort][i]['TP'][j]
				FN = confusion_matrix[cohort][i]['FN'][j]
				recall = TP / float((TP+FN))
				if recall >= recall_threshold:
					FP = confusion_matrix[cohort][i]['FP'][j]
					TN = confusion_matrix[cohort][i]['TN'][j]
					if calculation=='OR': # Odds Ratio: OR = (TP/FP) / (FN/TN)
						calculation_values.append((float(TP)/FP) / (float(FN)/TN))
					elif calculation=='RR': # Risk Ratio / Relative Risk: RR = (TP/(TP+FN)) / (FP/(FP+TN))
						calculation_values.append((float(TP)/(TP+FN)) / (float(FP)/(FP+TN)))
					elif calculation=='accuracy': # accuracy = (TP + TN) / (TP + TN + FP + FN)
						calculation_values.append(float(TP+TN) / (TP+FN+FP+TN))
					else: # precision = (TP) / (TP+FP)
						calculation_values.append(float(TP) / (TP+FP))
					break
		# Calculate average and variance of value of interest across all iterations for given cohort
		cohort_calculated_values_mean[cohort] = np.mean(calculation_values)
		cohort_calculated_values_var[cohort] = np.var(calculation_values)
	# Return table of average/variance values for performance on all cohorts at given threshold
	cohort_calculated_values_table = pd.concat([pd.Series(cohort_calculated_values_mean, name='Average '+calculation),
												pd.Series(cohort_calculated_values_var, name=calculation+' Var')], axis=1)
	print calculation, 'calculation completed for all cohorts', round(time.time()-runtime, 2), 'seconds.'
	if save_results == False:
		return cohort_calculated_values_table
	else:
		cohort_calculated_values_table.to_csv(outdir+calculation+'_results.csv')
		return cohort_calculated_values_table


