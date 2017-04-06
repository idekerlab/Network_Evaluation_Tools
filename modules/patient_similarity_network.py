###########################################################################
# ---------- Patient Similarity Network Construction Functions ---------- #
###########################################################################
import os
import pandas as pd
import networkx as nx
import time
import numpy as np
from sklearn.decomposition import PCA
import network_propagation as prop

# Load somatic mutation data (construct binary somatic mutation matrix in context of molecular network)
# If directory given: constructs combined somatic mutation matrix from all MAF files - Assumes filenames from Firehose
# Save path extention will be compressed matrix '.hdf'
def load_TCGA_MAF(MAF_path, save_path=None):
	print 'Loading Data...'
	if os.path.isdir(MAF_path):
		MAF_fn = [fn for fn in os.listdir(MAF_path) if fn.endswith('.maf.annotated')]
		for i in range(len(MAF_fn)):
			starttime = time.time()
			MAF_file_path = MAF_path + MAF_fn[i]
			cx_type = MAF_fn[i].split('-')[0]
			# Load MAF File
		    TCGA_MAF = pd.read_csv(MAF_file_path, sep='\t',low_memory=False)
		    # Convert mutation list to binary matrix format
		    sm_mat = TCGA_MAF.groupby(['Tumor_Sample_Barcode', 'Hugo_Symbol']).size().unstack().fillna(0)
		    sm_mat = (sm_mat>0).astype(int)
		    # Concat somatic mutation matrices together
		    if i==0:
		        TCGA_sm_mat = sm_mat
		    else:
		        TCGA_sm_mat = pd.concat([TCGA_sm_mat, sm_mat])
		    print cx_type, 'MAF loaded:', sm_mat.shape[0], 'patients,', sm_mat.shape[1], 'genes'
	else:
		# Load MAF File
	    TCGA_MAF = pd.read_csv(MAF_path, sep='\t',low_memory=False)
	    # Convert mutation list to binary matrix format
	    TCGA_sm_mat = TCGA_MAF.groupby(['Tumor_Sample_Barcode', 'Hugo_Symbol']).size().unstack().fillna(0)
	    TCGA_sm_mat = (TCGA_sm_mat>0).astype(int)
	# Trim TCGA barcodes
	TCGA_sm_mat.index = [pat[:12] for pat in TCGA_sm_mat.index]
	# Filter all patients completely with more than 1 patient ID from PanCancer Data
	sm_mat_filt = TCGA_sm_mat.ix[TCGA_sm_mat.index.value_counts().index[TCGA_sm_mat.index.value_counts()==1]].fillna(0)
	print 'TCGA somatic mutation data loaded:', time.time()-starttime, 'seconds.', sm_mat_filt.shape[0], 'patients,', sm_mat_filt.shape[1], 'genes'
	if save_path == None:
		return sm_mat_filt
	else:
		sm_mat_filt.to_hdf(save_path, key='SomaticMutationMatrix', mode='w')
		return sm_mat_filt

# Mean-centering of propagated somatic mutation profiles
def mean_center_data(propagated_sm_matrix):
	return propagated_sm_matrix - propagated_sm_matrix.mean()

# PCA reduction (up to threshold t of explained variance) of mean-centered, propagated somatic mutation profiles
def perform_PCA(propagated_sm_matrix, t=0.9):
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
def KNN_joining(similarity_df, k=5):
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

# Wrapper function for construction patient similarity network
def PSN_Constructor(network_path, MAF_path, outdir, delimiter='\t', min_mut=1, save_similarity=False):
	# Load network
	network_name = network_path.split('/')[-1].split('.')[0]
	network = prop.load_network_file(network_path, delimiter=delimiter)
	print 'Composite Network loaded:', network_path

	# Filter all TCGA patients with at least min_mut mutations in the network
	sm_mat = load_TCGA_MAF(MAF_path)
	network_mut_genes = list(set(network.nodes()).intersection(set(sm_mat.columns)))
	sm_mat_filt = sm_mat.ix[(sm_mat[network_mut_genes]>0).sum(axis=1)>=min_mut]

	# TCGA PanCan PSN
	print '---------- Constructing Patient Similarity Network ----------'
	# Propagate somatic mutation data
	prop_data = prop.closed_form_network_propagation(network, sm_mat_filt)
	# Mean center data
	prop_centered = mean_center_data(prop_data)
	# PCA Reduce centered data
	prop_centered_PCA = perform_PCA(prop_centered)
	# Pairwise Spearman on PCA reduced data
	prop_centered_PCA_spearman = pairwise_spearman(prop_centered_PCA)
	# Save pairwise patient similarity
	if save_similarity:
		prop_centered_PCA_spearman.to_hdf(outdir+network_name+'_PatSimilarities.hdf', key=network_name, mode='w')
		print network_name, 'patient similarity file saved'
	else:
		pass
	# Symmetric Z-normalization of pairwise spearman data
	prop_centered_PCA_spearman_znorm = symmetric_z_norm(prop_centered_PCA_spearman)
	# Construct PSN graph
	PSN = KNN_joining(prop_centered_PCA_spearman_znorm)
	# Write PSN to file
	nx.write_edgelist(PSN, outdir+network_name+'_PSN.txt', delimiter='\t', data=False)	
	print network_name, 'PSN Constructed:', len(PSN.nodes()), 'patients', len(PSN.edges()), 'edges'
	return PSN