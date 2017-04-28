#######################################################
# ---------- Network Propagation Functions ---------- #
#######################################################
import networkx as nx
import time
import numpy as np
import scipy
import pandas as pd
import copy
from numba import jit

# Normalize network (or network subgraph) for random walk propagation
def normalize_network(network):
	adj_mat = nx.adjacency_matrix(network)
	adj_array = np.array(adj_mat.todense())
	degree_norm_array = np.zeros(adj_array.shape)
	degree_sum = sum(adj_array)
	for i in range(len(degree_norm_array)):
		degree_norm_array[i,i]=1/float(degree_sum[i])
	sparse_degree_norm_array = scipy.sparse.csr_matrix(degree_norm_array)
	adj_array_norm = sparse_degree_norm_array.dot(adj_mat)
	return adj_array_norm
# Note about normalizing by degree, if multiply by degree_norm_array first (D^-1 * A), then do not need to return
# transposed adjacency array, it is already in the correct orientation

# Calculate optimal propagation coefficient
def calculate_alpha(network, m, b):
	avg_node_degree = np.log10(np.mean(network.degree().values()))
	alpha_val = round(m*avg_node_degree+b,3)
	if alpha_val <=0:
		raise ValueError('Alpha <= 0 - Network Avg Node Degree is too high')
		# There should never be a case where Alpha >= 1, as avg node degree will never be negative
	else:
		return alpha_val

# Closed form random-walk propagation (as seen in HotNet2) for each subgraph: Ft = (1-alpha)*Fo * (I-alpha*norm_adj_mat)^-1
@jit(nopython=True)
def fast_random_walk(alpha, binary_mat, adj_mat_norm):
    term1=(1-alpha)*binary_mat
    term2=np.identity(binary_mat.shape[1])-alpha*adj_mat_norm
    term2_inv = np.linalg.inv(term2)
    random_walk = np.dot(term1, term2_inv)
    return random_walk

# Wrapper for random walk propagation of full network by subgraphs
def closed_form_network_propagation(network, binary_matrix, alpha=None, m=-0.17190024, b=0.7674828, verbose=False, save_path=None):
	starttime=time.time()
	# Calculate alpha from network (resulting alpha must be <1)
	if alpha!=None:
		network_alpha=alpha
	else:
		network_alpha = calculate_alpha(network, m, b)
	if verbose:
		print 'Alpha:', network_alpha
	# Separate network into connected components and calculate propagation values of each sub-sample on each connected component
	subgraphs = list(nx.connected_component_subgraphs(network))
	prop_data_node_order = []
	for i in range(len(subgraphs)):
		subgraph = subgraphs[i]
		# Get nodes of subgraph
		subgraph_nodes = subgraph.nodes()
		prop_data_node_order = prop_data_node_order + subgraph_nodes
		# Filter binary_matrix by nodes of subgraph
		binary_matrix_filt = np.array(binary_matrix.T.ix[subgraph_nodes].fillna(0).astype(int).T)
		# Normalize each network subgraph for propagation
		subgraph_norm = normalize_network(subgraph)
		# Propagate and concatenate results 
		if i==0:
			prop_data = fast_random_walk(network_alpha, binary_matrix_filt, subgraph_norm.toarray())
		else:
			subgraph_Fn = fast_random_walk(network_alpha, binary_matrix_filt, subgraph_norm.toarray())
			prop_data = np.concatenate((prop_data, subgraph_Fn), axis=1)
	# Return propagated result as dataframe
	prop_data_df = pd.DataFrame(data=prop_data, index = binary_matrix.index, columns=prop_data_node_order)
	if save_path==None:
		if verbose:
			print 'Network Propagation Complete:', time.time()-starttime, 'seconds'		
		return prop_data_df
	else:
		prop_data_df.to_csv(save_path)
		if verbose:
			print 'Network Propagation Complete:', time.time()-starttime, 'seconds'				
		return prop_data_df

# Propagate binary matrix via iterative/power form of random walk model
def iterative_network_propagation(network, binary_matrix, max_iter=250, tol=1e-8, alpha=None, m=-0.17190024, b=0.7674828, verbose=False, save_path=None):
	starttime=time.time()
	# Calculate alpha
	if alpha!=None:
		network_alpha = alpha
	else:
		network_alpha = calculate_alpha(network, m, b)
	if verbose:
		print 'Alpha:', network_alpha
	# Normalize full network for propagation
	starttime = time.time()
	norm_adj_mat = normalize_network(network)
	if verbose:
	   print "Network Normalized", time.time()-starttime, 'seconds'
	# Initialize data structures for propagation
	Fi = scipy.sparse.csr_matrix(binary_matrix.T.ix[network.nodes()].fillna(0).astype(int).T)
	Fn_prev = copy.deepcopy(Fi)
	step_RMSE = [sum(sum(np.array(Fi.todense())))]
	# Propagate forward
	i = 0
	while (i <= max_iter) and (step_RMSE[-1] > tol):
		if i == 0:
			Fn = network_alpha*np.dot(Fi, norm_adj_mat)+(1-network_alpha)*Fi
		else:
			Fn_prev = Fn
			Fn = network_alpha*np.dot(Fn_prev, norm_adj_mat)+(1-network_alpha)*Fi
		step_diff = (Fn_prev-Fn).toarray().flatten()
		step_RMSE.append(np.sqrt(sum(step_diff**2) / len(step_diff)))
		i+=1
	prop_data_df = pd.DataFrame(data=Fn.todense(), index=binary_matrix.index, columns = network.nodes())
	if save_path==None:
		if verbose:
			print 'Network Propagation Complete:', i, 'steps,', time.time()-starttime, 'seconds, step RMSE:', step_RMSE[-1]		
		return prop_data_df, step_RMSE[1:]
	else:
		prop_data_df.to_csv(save_path)
		if verbose:
			print 'Network Propagation Complete:', i, 'steps,', time.time()-starttime, 'seconds, step RMSE:', step_RMSE[-1]				
		return prop_data_df, step_RMSE[1:]