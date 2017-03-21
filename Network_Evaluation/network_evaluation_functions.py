##################################################
# ---------- General Helper Functions ---------- #
##################################################
import networkx as nx

# Load network from file
def load_network_file(network_file_path, delimiter):
	network = nx.read_edgelist(input_network, delimiter=delimiter)
	return network

#######################################################
# ---------- Network Propagation Functions ---------- #
#######################################################
import time
import numpy as np
import scipy
import copy

# Normalize network (or network subgraph) for random walk propagation
def normalize_network(network):
	starttime = time.time()
	adj_mat = nx.adjacency_matrix(network)
	adj_array = np.array(adj_mat.todense())
	degree_norm_array = np.zeros(adj_array.shape)
	degree_sum = sum(adj_array)
	for i in range(len(degree_norm_array)):
		degree_norm_array[i,i]=1/float(degree_sum[i])
	sparse_degree_norm_array = scipy.sparse.csr_matrix(degree_norm_array)
	adj_array_norm = np.dot(sparse_degree_norm_array, adj_mat)
	print "Subgraph Normalized", time.time()-starttime, 'seconds'
	return adj_array_norm
# Note about normalizing by degree, if multiply by degree_norm_array first (D^-1 * A), then do not need to return
# transposed adjacency array, it is already in the correct orientation

# Calculate optimal propagation coefficient
def calculate_alpha(network):
	m, b = -0.17190024, 0.7674828
	avg_node_degree = np.log10(np.mean(network.degree().values()))
	alpha_val = round(m*avg_node_degree+b,3)
	if alpha_val <=0:
		raise ValueError('Alpha <= 0 - Network Avg Node Degree is too high')
		# There should never be a case where Alpha >= 1, as avg node degree will never be negative
	else:
		return alpha_val

# Propagate binary matrix via closed form of random walk model
def closed_form_network_propagation(network, binary_node_sets_matrix):
	starttime=time.time()
	# Calculate alpha from network (resulting alpha must be <1)
	network_alpha = calculate_alpha(network)
	# Separate network into connected components and calculate propagation values of each sub-sample on each connected component
		# Normalize network for propagation
		norm_adj_mat = normalize_network(network)
		# Closed form random-walk propagation (as seen in HotNet2)
		# Ft = (1-alpha)*Fo * (I-alpha*norm_adj_mat)^-1
		term1=(1-network_alpha)*binary_node_sets_matrix
		term2=np.identity(binary_node_sets_matrix.shape[0])-network_alpha*norm_adj_mat
		term2_inv = np.linalg.inv(term2)
		# Ft = term1 * term2^-1
		prop_data = np.dot(term1, term2_inv)
	# Concatenate results 


	print 'Closed Propagation:', time.time()-starttime, 'seconds'
	return prop_data

# Propagate binary matrix via iterative/power form of random walk model
def iterative_network_propagation(network, binary_node_sets_matrix, max_iter=250, tol=1e-4):
	starttime=time.time()
	# Calculate alpha
	network_alpha = calculate_alpha(network)
	# Normalize full network for propagation
	norm_adj_mat = normalize_network(network)
	# Initialize data structures for propagation
	Fn = np.array(binary_node_sets_matrix[network.nodes()])
    Fn_prev = copy.deepcopy(Fn)
    step_RMSE = [sum(sum(copy.deepcopy(Fn)))]
	# Propagate forward
	while (i <= max_iter) and (step_RMSE > tol):
    
    #Propagate forward
    Fn=np.array(binary_node_sets_matrix[])
    Fn_prev = np.array(binary_node_sets_matrix)
    prop_geno_RMSE=[sum(sum(np.array(Fo)))]
    for i in range(num_iter):
        if i==0:
            Fn = alpha*(sm_mat.dot(adj_mat))+(1-alpha)*Fi
            #Fn=prop_step(norm_adj_mat,Fn,np.array(Fo),alpha)
        else:
            Fn_prev=Fn
            Fn=prop_step(norm_adj_mat,Fn,np.array(Fo),alpha)
        step_RMSE = np.sqrt(sum(sum((Fn_prev-Fn)**2)))/norm_adj_mat.shape[0]
        if step_RMSE == prop_geno_RMSE[-1]:
            prop_geno_RMSE.append(step_RMSE)
            break
        else:
            prop_geno_RMSE.append(step_RMSE)
    else:
    	print 'Max Iterations Reached'
    print 'Iterative Propagation:', time.time()-starttime3, 'seconds'
    return Fn


###########################################################################
# ---------- Patient Similarity Network Construction Functions ---------- #
###########################################################################
import pandas as pd
from sklearn.decomposition import PCA

# Load somatic mutation data (construct binary somatic mutation matrix in context of molecular network)
def load_TCGA_MAF(MAF_file):
	# Process MAF file to produce dictionary of patient mutations
    TCGA_MAF = pd.read_csv(maf_file,sep='\t',low_memory=False)
    # Convert to matrix format
    TCGA_sm_mat = TCGA_MAF.groupby(['Tumor_Sample_Barcode', 'Hugo_Symbol']).size().unstack().fillna(0)
    # Save only as int
    TCGA_sm_mat = (TCGA_sm_mat>0).astype(int)
    # Trim TCGA barcodes
    TCGA_sm_mat.index = [pat[:12] for pat in TCGA_sm_mat.index]
    return TCGA_sm_mat

# Mean-centering of propagated somatic mutation profiles
def mean_center_data(propagated_sm_matrix):
	return propagated_sm_matrix - propagated_sm_matrix.mean()

# PCA reduction (up to threshold t of explained variance) of mean-centered, propagated somatic mutation profiles
def perform_PCA(propagated_sm_matrix, t):
	starttime = time.time()
	# Construct PCA model
	pca = PCA()
	# Transform propagated somatic mutation profiles by PCA
	fit_transform_res = pca.fit_transform(propagated_sm_matrix)
	propagated_profile_pca = pd.DataFrame(data=fit_transform_res, index=propagated_sm_matrix.index)
	PCs=['PC'+repr(i+1) for i in propagated_profile_pca.columns]
	propagated_profile_pca.columns = PCs
	# Calculate the variance explained by each PC
	explained_variance_ratio = pca.explained_variance_ratio_
	# Reduce propagated somatic mutation profiles to capture a specific proportion of the data's variance (t)
	explained_variance_cumsum = 0
	for i in range(len(explained_variance_ratio)):
		explained_variance_cumsum = explained_variance_cumsum + explained_variance_ratio[i]
		if explained_variance_cumsum >= t:
			print 'PCA complete:', time.time()-starttime, 'seconds. Precise explained variance:', explained_variance_cumsum
			return propagated_profile_pca.iloc[:,:i]

# Pairwise spearman correlation of PCA reduced, mean-centered, propagated somatic mutation profile
def pairwise_spearman(propagated_sm_matrix):
	starttime = time.time()
	# Convert rows of PCA reduced patient profiles to rankings and change to array
	data_rank_df = data_df.rank(axis=1)
	data_rank_array = data_rank_df.as_matrix()
	# Fast pearson correlation calculation on ranks
	data_rank_array_dot = data_rank_array.dot(data_rank_array.T)
	e_xy = data_rank_array_dot / data_rank_array.shape[1]
	data_rank_array_means = np.asarray(data_rank_array.mean(1)).flatten()
	e_x_e_y = np.asarray(np.matrix(data_rank_array_means.reshape(-1, 1)) * np.matrix(data_rank_array_means.reshape(1, -1)))
	corr = e_xy - e_x_e_y
	std = np.sqrt(np.asarray((data_rank_array * data_rank_array).sum(1)).flatten() / data_rank_array.shape[1] - np.power(np.asarray(data_rank_array.mean(1)).flatten(), 2))
	corr = corr / std.reshape(1, -1)
	corr = corr / std.reshape(-1, 1)
	# Convert pairwise correlation array to dataframe
	spearman_corr_df = pd.DataFrame(corr, index=data_df.index, columns=data_df.index)
	print 'Pairwise correlation calculation complete:', time.time()-starttime, 'seconds'
	return spearman_corr_df

# Symmetric Z-normalization of pairwise correlation matrix
def symmetric_z_norm(similarity_df):
	starttime = time.time()
	# Row Z-normalization
	similarity_df_rows = similarity_df.subtract(similarity_df.mean()).divide(similarity_df.std())
	# Column Z-normalization (transpose of row Z-normalization due to input symmetry)
	similarity_df_cols = similarity_df_rows.T
	# Average Row and Column Z-normlizations of table
	symmetric_z_table = (similarity_df_rows + similarity_df_cols) / 2
	print 'Symmetric Z normalization of similarity matrix complete:', time.time()-starttime, 'seconds'
	return symmetric_z_table

# Conversion of normalized pairwise similarities to actual network by using KNN and top k similarities for each node
def KNN_joining(similarity_df, k):
	starttime = time.time()
	pairwise_sim_array = np.array(similarity_df)
	np.fill_diagonal(pairwise_sim_array, -1)
	diag_adjust_pairwise_sim_array = pd.DataFrame(pairwise_sim_array, index = similarity_df.index, columns = similarity_df.columns)
	G = nx.Graph()
	for pat in diag_adjust_pairwise_sim_array.index:
		pat_knn = diag_adjust_pairwise_sim_array.ix[pat].sort_values(ascending=False).ix[:k].index
		for neighbor in pat_knn:
			G.add_edge(pat, neighbor)
	print 'Network construction complete:', time.time()-starttime, 'seconds'
	return G

################################################################
# ---------- Set-Based Network Evaluation Functions ---------- #
################################################################
import random
import scipy.stats as stats
import sklearn.metrics as metrics

# Shuffle network in degree-preserving manner
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
			node_set_sub_sample = random.sample(intersect, sample_size) for i in range(n)
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
























