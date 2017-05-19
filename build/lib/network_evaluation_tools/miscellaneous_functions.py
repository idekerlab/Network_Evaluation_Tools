import time
import pandas as pd
import numpy as np
import data_import_tools as dit
import network_propagation as prop
import network_evaluation_functions as nef
from multiprocessing import Pool
import pickle as p

################################################################################
# ---------- Additional Node Set-Based Network Evaluation Functions ---------- #
################################################################################

# Calculate confusion matrix (true positive, false negatives, false positives, true negatives) of node set recovery for given node set
# The confusion matrix for every position on every AUPRC curve is returned/stored
def calculate_confusion_matrix_serial(prop_geno, p, n, node_set_name, node_set, verbose=False):
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
	if verbose:
		print 'Confusion matrices calculated for node set', node_set_name, 'complete.', repr(len(intersect))+' nodes in network,', round(time.time()-runtime, 2), 'seconds.'
	return confusion_matrices

# Calculate confusion matrix (true positive, false negatives, false positives, true negatives) of node set recovery for given node set 
# The parameter setup is written for running in serial, only difference is that the name of the node set also must be passed, and prop_geno will be set as a global variable
# The confusion matrix for every position on every AUPRC curve is returned/stored
def calculate_confusion_matrix_parallel(node_set_params):
	node_set_name, node_set, p, n, verbose = node_set_params[0], node_set_params[1], node_set_params[2], node_set_params[3], node_set_params[4]
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
	if verbose:
		print 'Confusion matrices calculated for node set', node_set_name, 'complete.', repr(len(intersect))+' nodes in network,', round(time.time()-runtime, 2), 'seconds.'
	return [node_set_name, confusion_matrices]

# Wapper for calculating the confusion matrices for input node set file and network (has parallel option)
# Not run for null network shuffles
def confusion_matrix_construction_wrapper(network_file, node_set_file, sample_p, sub_sample_iterations, 
	alpha=None, m=-0.17190024, b=0.7674828, net_delim='\t', set_delim='\t', cores=1, verbose=False, save_path=None):
	starttime = time.time()
	# Load network
	network = dit.load_network_file(network_file, delimiter=net_delim, verbose=verbose)
	# Load node set
	node_sets = dit.load_node_sets(node_set_file, delimiter=set_delim, verbose=verbose)
	# Calculate network influence matrix
	prop_net = nef.construct_prop_kernel(network, alpha=alpha, m=m, b=b)
	# Calculate confusion matrix values for each node set
	if cores == 1:
		# Calculate confusion matrix values for node sets one at a time
		node_set_conf_mat = {node_set:nef.calculate_confusion_matrix_serial(prop_net, sample_p, sub_sample_iterations, node_set, node_sets[node_set], verbose=verbose) for node_set in node_sets}
	else:
		# Initialize multiple threads for confusion matrix analysis of multiple node sets
		initializer_args = [prop_net]
		pool = Pool(cores, nef.parallel_analysis_initializer, initializer_args)
		# Construct parameter list to be passed
		conf_mat_Analysis_params = [[node_set, node_sets[node_set], sample_p, sub_sample_iterations, verbose] for node_set in node_sets]
		# Run the confusion matrix analysis for each geneset
		conf_mat_results = pool.map(nef.calculate_confusion_matrix_parallel, conf_mat_Analysis_params)
		# Construct confusion matrix results dictionary
		node_set_conf_mat = {result[0]:result[1] for result in conf_mat_results}
	if save_path == None:
		if verbose:
			print 'Network confusion matrix values calcualted:', round(time.time()-starttime, 2), 'seconds'			
		return node_set_conf_mat
	else:
		p.dump(node_set_conf_mat, open(save_path, 'wb'))
		if verbose:
			print 'Network confusion matrix values calcualted:', round(time.time()-starttime, 2), 'seconds'					
		return node_set_conf_mat

# Use confusion matrix results to calculate odds ratio, risk ratio, accuracy or precision at a given recall threshold
def confusion_matrix_analysis(confusion_matrix_input, calculation, recall_threshold=0.9, verbose=False, save_path=None):
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
	if save_path == None:
		if verbose:
			print calculation, 'calculation completed for all cohorts', round(time.time()-runtime, 2), 'seconds.'
		return cohort_calculated_values_table
	else:
		cohort_calculated_values_table.to_csv(save_path)
		if verbose:
			print calculation, 'calculation completed for all cohorts', round(time.time()-runtime, 2), 'seconds.'		
		return cohort_calculated_values_table


