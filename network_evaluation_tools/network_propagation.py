#######################################################
# ---------- Network Propagation Functions ---------- #
#######################################################
import networkx as nx
import time
import numpy as np
import scipy
import pandas as pd
import copy


# Normalize network (or network subgraph) for random walk propagation
def normalize_network(network, symmetric_norm=False):
    adj_mat = nx.adjacency_matrix(network)
    adj_array = np.array(adj_mat.todense())
    if symmetric_norm:
        D = np.diag(1 / np.sqrt(sum(adj_array)))
        adj_array_norm = np.dot(np.dot(D, adj_array), D)
    else:
        degree_norm_array = np.diag(1 / sum(adj_array).astype(float))
        sparse_degree_norm_array = scipy.sparse.csr_matrix(degree_norm_array)
        adj_array_norm = sparse_degree_norm_array.dot(adj_mat).toarray()
    return adj_array_norm


# Note about normalizing by degree, if multiply by degree_norm_array first (D^-1 * A), then do not need to return
# transposed adjacency array, it is already in the correct orientation

# Calculate optimal propagation coefficient (updated model)
def calculate_alpha(network, m=-0.02935302, b=0.74842057):
    log_edge_count = np.log10(len(network.edges()))
    alpha_val = round(m * log_edge_count + b, 3)
    if alpha_val <= 0:
        raise ValueError('Alpha <= 0 - Network Edge Count is too high')
    # There should never be a case where Alpha >= 1, as avg node degree will never be negative
    else:
        return alpha_val


# Closed form random-walk propagation (as seen in HotNet2) for each subgraph: Ft = (1-alpha)*Fo * (I-alpha*norm_adj_mat)^-1
# Concatenate to previous set of subgraphs
def fast_random_walk(alpha, binary_mat, subgraph_norm, prop_data):
    term1 = (1 - alpha) * binary_mat
    term2 = np.identity(binary_mat.shape[1]) - alpha * subgraph_norm
    term2_inv = np.linalg.inv(term2)
    subgraph_prop = np.dot(term1, term2_inv)
    return np.concatenate((prop_data, subgraph_prop), axis=1)


# Wrapper for random walk propagation of full network by subgraphs
def closed_form_network_propagation(network, binary_matrix, network_alpha, symmetric_norm=False, verbose=False,
                                    save_path=None):
    starttime = time.time()
    if verbose:
        print 'Alpha:', network_alpha
    # Separate network into connected components and calculate propagation values of each sub-sample on each connected component
    subgraphs = list(nx.connected_component_subgraphs(network))
    # Initialize propagation results by propagating first subgraph
    subgraph = subgraphs[0]
    subgraph_nodes = list(subgraph.nodes)
    prop_data_node_order = list(subgraph_nodes)
    binary_matrix_filt = np.array(binary_matrix.T.ix[subgraph_nodes].fillna(0).T)
    subgraph_norm = normalize_network(subgraph, symmetric_norm=symmetric_norm)
    prop_data_empty = np.zeros((binary_matrix_filt.shape[0], 1))
    prop_data = fast_random_walk(network_alpha, binary_matrix_filt, subgraph_norm, prop_data_empty)
    # Get propagated results for remaining subgraphs
    for subgraph in subgraphs[1:]:
        subgraph_nodes = list(subgraph.nodes)
        prop_data_node_order = prop_data_node_order + subgraph_nodes
        binary_matrix_filt = np.array(binary_matrix.T.ix[subgraph_nodes].fillna(0).T)
        subgraph_norm = normalize_network(subgraph, symmetric_norm=symmetric_norm)
        prop_data = fast_random_walk(network_alpha, binary_matrix_filt, subgraph_norm, prop_data)
    # Return propagated result as dataframe
    prop_data_df = pd.DataFrame(data=prop_data[:, 1:], index=binary_matrix.index, columns=prop_data_node_order)
    if save_path is None:
        if verbose:
            print 'Network Propagation Complete:', time.time() - starttime, 'seconds'
        return prop_data_df
    else:
        prop_data_df.to_csv(save_path)
        if verbose:
            print 'Network Propagation Complete:', time.time() - starttime, 'seconds'
        return prop_data_df
