###########################################################################
# ---------- Patient Similarity Network Construction Functions ---------- #
###########################################################################
import os
import pandas as pd
import networkx as nx
import time
import numpy as np
from sklearn.decomposition import PCA
import data_import_tools as dit
import network_propagation as prop

# Mean-centering of propagated somatic mutation profiles
def mean_center_data(propagated_sm_matrix, verbose=False):
	if verbose:
		print 'Data mean-centered'
	return propagated_sm_matrix - propagated_sm_matrix.mean()

# PCA reduction (up to threshold t of explained variance) of mean-centered, propagated somatic mutation profiles
def perform_PCA(propagated_sm_matrix, t=0.9, verbose=False):
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
			if verbose:
				print 'PCA complete:', time.time()-starttime, 'seconds. Precise explained variance:', explained_variance_cumsum
			return propagated_profile_pca.iloc[:,:i]

# Pairwise spearman correlation of PCA reduced, mean-centered, propagated somatic mutation profile
def pairwise_spearman(propagated_sm_matrix, verbose=False, save_path=None):
	starttime = time.time()
	# Convert rows of PCA reduced patient profiles to rankings and change to array
	data_rank_df = propagated_sm_matrix.rank(axis=1)
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
	spearman_corr_df = pd.DataFrame(corr, index=propagated_sm_matrix.index, columns=propagated_sm_matrix.index)
	if verbose:
		print 'Pairwise correlation calculation complete:', time.time()-starttime, 'seconds'
	if save_path==None:
		return spearman_corr_df
	else:
		spearman_corr_df.to_csv(save_path, key=network_name, mode='w')
		return spearman_corr_df

# Symmetric Z-normalization of pairwise correlation matrix
def symmetric_z_norm(similarity_df, verbose=False):
	starttime = time.time()
	# Row Z-normalization
	similarity_df_rows = similarity_df.subtract(similarity_df.mean()).divide(similarity_df.std())
	# Column Z-normalization (transpose of row Z-normalization due to input symmetry)
	similarity_df_cols = similarity_df_rows.T
	# Average Row and Column Z-normlizations of table
	symmetric_z_table = (similarity_df_rows + similarity_df_cols) / 2
	if verbose:
		print 'Symmetric Z normalization of similarity matrix complete:', time.time()-starttime, 'seconds'
	return symmetric_z_table

# Conversion of normalized pairwise similarities to actual network by using KNN and top k similarities for each node
def KNN_joining(similarity_df, k=5, verbose=False, save_path=None):
	starttime = time.time()
	pairwise_sim_array = np.array(similarity_df)
	np.fill_diagonal(pairwise_sim_array, -1)
	diag_adjust_pairwise_sim_array = pd.DataFrame(pairwise_sim_array, index = similarity_df.index, columns = similarity_df.columns).astype(float)
	G = nx.Graph()
	for pat in diag_adjust_pairwise_sim_array.index:
		pat_knn = diag_adjust_pairwise_sim_array.ix[pat].sort_values(ascending=False).ix[:k].index
		for neighbor in pat_knn:
			G.add_edge(pat, neighbor)
	if verbose:
		print 'Network construction complete:', time.time()-starttime, 'seconds', len(G.nodes()), 'patients', len(G.edges()), 'edges'
	if save_network==None:
		return G
	else:
		nx.write_edgelist(G, save_network, delimiter='\t', data=False)
		return G

# Wrapper function for construction patient similarity network
# binary_mut_mat_path can be a single file or directory, if directory given, it will concatenate all files in that directory
def PSN_Constructor(network_path, binary_mut_mat_path, net_delim='\t', mut_mat_filetype='matrix', mut_mat_delim='\t', min_mut=1, verbose=False,
					save_propagation=None, save_similarity=None, save_network=None):
	# Load network
	network = dit.load_network_file(network_path, delimiter=net_delim, verbose=verbose)

	# Filter all TCGA patients with at least min_mut mutations in the network
	if os.path.isdir(binary_mut_mat_path):
		filename_list = [binary_mut_mat_path+fn for fn in os.listdir(binary_mut_mat_path)]
		sm_mat = dit.concat_binary_matrices(filename_list, filetype=mut_mat_filetype, delimiter=mut_mat_delim)
	else:
		sm_mat = dit.load_binary_mutation_matrix(binary_mut_mat_path, filetype=mut_mat_filetype, delimiter=mut_mat_delim)
	network_mut_genes = list(set(network.nodes()).intersection(set(sm_mat.columns)))
	sm_mat_filt = sm_mat.ix[(sm_mat[network_mut_genes]>0).sum(axis=1)>=min_mut]
	if verbose:
		print 'Patients with <', min_mut, 'mutations in network removed:', sm_mat.shape[0] - sm_mat_filt.shape[0]

	# TCGA PanCan PSN
	if verbose:
		print '---------- Constructing Patient Similarity Network ----------'
	# Propagate somatic mutation data
	if save_propagation==None:
		prop_data = prop.closed_form_network_propagation(network, sm_mat_filt, verbose=verbose)
	else:
		prop_data = prop.closed_form_network_propagation(network, sm_mat_filt, verbose=verbose, save_path=save_propagation)
	# Mean center data
	prop_centered = mean_center_data(prop_data, verbose=verbose)
	# PCA Reduce centered data
	prop_centered_PCA = perform_PCA(prop_centered, verbose=verbose)
	# Pairwise Spearman on PCA reduced data
	if save_similarity==None:
		prop_centered_PCA_spearman = pairwise_spearman(prop_centered_PCA, verbose=verbose)
	else:
		prop_centered_PCA_spearman = pairwise_spearman(prop_centered_PCA, verbose=verbose, save_path=save_similarity)
	# Symmetric Z-normalization of pairwise spearman data
	prop_centered_PCA_spearman_znorm = symmetric_z_norm(prop_centered_PCA_spearman, verbose=verbose)
	# Construct PSN graph
	if save_network==None:
		PSN = KNN_joining(prop_centered_PCA_spearman_znorm, verbose=verbose)
	else:
		PSN = KNN_joining(prop_centered_PCA_spearman_znorm, verbose=verbose, save_path=save_network)
	return PSN