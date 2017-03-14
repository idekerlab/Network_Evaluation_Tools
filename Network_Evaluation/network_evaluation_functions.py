import networkx as nx
import numpy as np

##################################################
# ---------- General Helper Functions ---------- #
##################################################

# Load network from file
def load_network_file(network_file_path, delimiter):
	network = nx.read_edgelist(input_network, delimiter=delimiter)
	return network

#######################################################
# ---------- Network Propagation Functions ---------- #
#######################################################

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
	return np.array(adj_array_norm.todense())
# Note about normalizing by degree, if multiply by degree_norm_array first (D^-1 * A), then do not need to return
# transposed adjacency array, it is already in the correct orientation

# Calculate optimal propagation coefficient
def calculate_alpha(network):
	m, b = -0.17190024, 0.7674828
	avg_node_degree = np.log10(np.mean(network.degree().values()))
	alpha_val = round(m*avg_node_degree+b,3)
	return alpha_val

# Propagate binary matrix (each row is different node set to propagate)
def network_propagation(network, binary_node_sets_matrix, alpha, method='power'):

###########################################################################
# ---------- Patient Similarity Network Construction Functions ---------- #
###########################################################################

# Load somatic mutation data (construct binary somatic mutation matrix in context of molecular network)
def load_sm_data(sm_data_file):


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
def PSN_construction(similarity_df, k):
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

# PSN Construction wrapper

################################################################
# ---------- Set-Based Network Evaluation Functions ---------- #
################################################################

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

# Load node sets for SBNE (construct binary matrix for propagation in context of network for SBNE)
def load_SBNE_node_sets(node_set_file, network):
	# Construct node set sub-sample matrices

# AUPRC Analysis for each node set

# Calculate null AUPRC distribution

# Calculate robust z-score for each AUPRC

# SBNE Wrapper function
























