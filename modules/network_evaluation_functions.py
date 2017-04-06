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

# Construct node-set sub-sampled binary matrix for propagation in context of network for SBNE with given sub-sample proportion p, and sub-sample number of iterations n
def SBNE_binary_matrix_constructor(node_set_file, network, p, n):
	# Construct node set sub-sample matrices
	f = open(node_set_file)
	node_set_lines = f.read().splitlines()
	node_set_lines_split = [line.split('\t') for line in node_set_lines]
	f.close()
	node_sets = {node_set[0]:set(node_set[1:]) for node_set in node_set_lines_split}
	node_set_names = sorted(node_sets.keys())
	# Initialize dataframe for node set sub-samples to be propagated
	network_nodes = set(network.nodes())
	node_set_sub_sample_binary_matrix_index = []
	for node_set in node_set_names:
		node_set_index = [node_set+'_'+repr(i) for i in range(1,n+1)]
		node_set_sub_sample_binary_matrix_index = node_set_sub_sample_binary_matrix_index + node_set_index
	node_set_sub_sample_binary_matrix = pd.DataFrame(0, index = node_set_sub_sample_binary_matrix_index, columns = list(network_nodes))
	# Sub-sample the intersection of each node set with the network node set
	for node_set in node_set_names:
		intersect = list(node_sets[node_set].intersection(network_nodes))
		sample_size = int(round(p*len(intersect)))
		for i in range(1, n+1):
			node_set_sub_sample = random.sample(intersect, sample_size)
			node_set_sub_sample_binary_matrix.ix[node_set+'_'+repr(i)][node_set_sub_sample] = 1
	return node_sets, node_set_sub_sample_binary_matrix

# AUPRC Analysis for each node set using labelled propagated node set sub-sample dataframe 
def AUPRC_Analysis(node_set_sub_sample_binary_matrix, node_sets):
	runtime = time.time()
	# Propagate sub-sampled node sets
	prop_sub_samples = network_propagation(network, node_set_sub_sample_binary_matrix)
	# AUPRC Analysis
	node_set_names = sorted(node_sets.keys())
	n_sub_samples = prop_sub_samples.shape[0] / len(node_set_names)
	node_set_AUPRC_results = []
	for node_set in node_set_names:
		node_set_AUPRCs = []
		for i in range(1, n_sub_samples+1):
			node_set_index = node_set+'_'+repr(i)
			# Get sub-sampled nodes
			sample = set(node_set_sub_sample_binary_matrix.ix[node_set_index][node_set_sub_sample_binary_matrix.ix[node_set_index]==1].index)
			# Get remaining nodes in node set not sub-sampled
			node_set_non_sample = node_sets[node_set].difference(sample)
			# Sort all non-sampled_genes by propagation score
			sorted_nodes = prop_sub_samples.ix[node_set_index][prop_sub_samples.columns[~prop_sub_samples.columns.isin(sample)]].sort(ascending=False, inplace=False).index
			# Binary vector of sorted genes by propagation score marking non-sampled node set nodes
			y_actual = [1 if node in node_set_non_sample else 0 for node in sorted_nodes]
			# Identify points on the node list to calculate precision/recall
			sorted_node_set_non_sample_index = [j+1 for j in range(len(y_actual)) if y_actual[j]==1]
			# Construct precision-recall curve and calculate AUPRC
			precision, recall = [], []
			for k in sorted_node_set_non_sample_index:
			    TP, FN =  sum(y_actual[:k]), sum(y_actual[k:])
			    precision.append(TP/float(k))
			    recall.append(TP/float(TP+FN))
			precision = [1]+precision+[0]
			recall=[0]+recall+[1]
			node_set_AUPRCs.append(metrics.auc(recall, precision))
		# Average and Variance of AUPRCs for across all sub-samplings of node set
		node_set_AUPRC_results = [node_set, np.mean(node_set_AUPRCs), np.var(node_set_AUPRCs)]
	return pd.DataFrame(node_set_AUPRC_results, columns = ['Node Set Name', 'Avg AUPRC', 'AUPRC Var']).set_index('Node Set Name')

# Calculate robust z-score for each AUPRC (requires network, and SBNE node sub-sampling matrix)
def calculate_robust_z(network, binary_node_sub_sample_matrix, n_shuffles):
		# Calculate AUPRC result of node sets on actual network
	actualnet_AUPRC_result = AUPRC_Analysis(network, binary_node_sub_sample_matrix)['Avg AUPRC']

	# Calculate AUPRC results of node set on shuffled networks
	shuffnet_AUPRCs = []
	for i in range(n_shuffles):
		# Shuffle network
		shuffled_network = shuffle_network(network)
		# Calculate AUPRC Results
		shuffnet_AUPRCs.append(AUPRC_Analysis(shuffled_network, binary_node_sub_sample_matrix)['Avg AUPRC'])
	# Concatinate shuffled network AUPRC results together
	shuffnet_AUPRC_table = pd.concat([pd.Series(shuffnet_AUPRCs, name="Shuffled_Network_"+repr(i+1)) for i in range(n_shuffles)], axis =1)

	# Calculate robust z-score of network for each node set
	# Mean absolute deviation scaling factor to make median absolute deviation behave similarly to the standard deviation of a normal distribution
	k = 1/stats.norm.ppf(0.75)
	# Compute robust z-score for composite network performances
	AUPRC_null_median = shuffnet_AUPRC_table.median(axis=1)
	AUPRC_null_MAD = abs(shuffnet_AUPRC_table.subtract(AUPRC_null_median, axis=0)).median(axis=1)
	AUPRC_null_MAD_scaled = k*AUPRC_null_MAD
	AUPRC_ZNorm = (actualnet_AUPRC_result - AUPRC_null_median).divide(AUPRC_null_MAD_scaled)
	return AUPRC_ZNorm
























