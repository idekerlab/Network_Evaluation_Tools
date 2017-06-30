################################################################
# ---------- Network-Based Stratification Functions ---------- #
################################################################

############################################
# ---------- Core NBS Functions ---------- #
############################################
import random
import networkx as nx
import pandas as pd
import scipy.stats as stats
import numpy as np
import numpy.matlib as matlib
from scipy.optimize import nnls
import time

# Function to construct the KNN regularization network graph laplacian
# network is a NetworkX object
# gamma is the Vandin 2011 diagonal correction value (should be small)
# kn is the number of nearest neighbors to construct network regularizer from
def network_inf_KNN_glap(network, gamma=0.01, kn=11, verbose=False, save_path=None):
    glap_inv_starttime = time.time()
    # Construct network laplacian matrix
    network_nodes = network.nodes()
    L_arr = nx.laplacian_matrix(network).todense()
    # Adjust diagonal of laplacian matrix by small gamma as seen in Vandin 2011
    L_vandin = L_arr + gamma*np.identity(len(network_nodes))
    # Calculate the inverse of diagonal adjusted graph laplacian to get graph influence matrix (re: Vandin 2011)
    # This is significantly faster than the method proposed previously in NBS v0.2.0 to calculate the pseudoinverse
    # of each network component. The graph result may be slightly different, and thus the resulting clustering results.
    # But our analysis suggest that the results are not affected greatly (via OV on HM90 task.)
    L_inv_arr = np.linalg.inv(L_vandin)
    L_inv = pd.DataFrame(L_inv_arr, index = network_nodes, columns = network_nodes)
    if verbose:
        print 'Graph influence matrix calculated:', time.time()-glap_inv_starttime, 'seconds'
    KNN_starttime = time.time()
    # Construct KNN graph using the 11 nearest neighbors by influence score (glap_pinv)
    # The code may include each gene's self as the 'nearest' neighbor
    # Therefore the resulting laplacian maybe more like KNN with k=10 with self-edges
    # We only draw edges where the influence is > 0 between nodes
    KNN_graph = nx.Graph()
    for gene in L_inv.index:
        gene_knn = L_inv.ix[gene].sort_values(ascending=False)[:kn].index
        for neighbor in gene_knn:
            if L_inv.ix[gene][neighbor] > 0:
                KNN_graph.add_edge(gene, neighbor)
    KNN_nodes = KNN_graph.nodes()
    # Calculate KNN graph laplacian
    knnGlap_sparse = nx.laplacian_matrix(KNN_graph)
    knnGlap = pd.DataFrame(knnGlap_sparse.todense(), index=KNN_nodes, columns=KNN_nodes)
    if save_path is not None:
        knnGlap.to_hdf(save_path, key='knnGLap', mode='w')     
    if verbose:
        print 'Graph laplacian of KNN network from influence matrix calculated:', time.time()-KNN_starttime, 'seconds'    
    return knnGlap

# Function to sub-sample binary somatic mutation profile data frame in context of a given network
# If no network (propNet) is given, all genes are sub-sampled
# Key is that filtering for min mutation count happens before filtering by network nodes not after
def subsample_sm_mat(sm_mat, propNet=None, pats_subsample_p=0.8, gene_subsample_p=0.8, min_muts=10):
    # Number of indiv/features for sampling
    (Nind, Dfeat) = sm_mat.shape
    Nsample = round(Nind*pats_subsample_p)
    Dsample = round(Dfeat*gene_subsample_p)

    # Sub sample patients
    pats_subsample = random.sample(sm_mat.index, int(Nsample))
    # Sub sample genes
    gene_subsample = random.sample(sm_mat.columns, int(Dsample))
    # Sub sampled data mat
    gind_sample = sm_mat.ix[pats_subsample][gene_subsample]
    # Filter by mutation count
    gind_sample = gind_sample[gind_sample.sum(axis=1) > min_muts]
    if propNet is not None:
        # Filter columns by network nodes only (if network is given)
        gind_sample_filt = gind_sample.T.ix[propNet.nodes()].fillna(0).T
        return gind_sample_filt
    else:
        return gind_sample

# Function to sub-sample binary somatic mutation profile data frame in context of a given network
# If no network (propNet) is given, all genes are sub-sampled
def subsample_sm_mat_old(sm_mat, propNet=None, pats_subsample_p=0.8, gene_subsample_p=0.8, min_muts=10):
    # Sub sample patients
    pats_subsample = random.sample(sm_mat.index, int(round(len(sm_mat.index)*pats_subsample_p)))
    # Sub sample genes in network
    if type(propNet)==nx.Graph:
        propNet_nodes = propNet.nodes()
        sm_network_genes = [gene for gene in propNet_nodes if gene in sm_mat.columns]
        gene_subsample = random.sample(sm_network_genes, int(round(len(sm_network_genes)*gene_subsample_p)))
    else:
        gene_subsample = random.sample(sm_mat.columns, int(round(len(sm_mat.columns)*gene_subsample_p)))
   
    # Sub sample binary matrix and keep only patients with >=10 Mutations
    sm_mat_subsample = sm_mat.ix[pats_subsample][gene_subsample]
    sm_mat_subsample_filt = sm_mat_subsample[sm_mat_subsample.sum(axis=1)>=min_muts]
    
    # Add empty columns for all genes in network not mutated in subsample
    if type(propNet)==nx.Graph:
        no_sm_network_genes = [gene for gene in propNet_nodes if gene not in sm_mat_subsample_filt.columns]
        empty_binary_sm_cols = pd.DataFrame(0.0, index=sm_mat_subsample_filt.index, columns=no_sm_network_genes)
        sm_mat_subsample_full = pd.concat([sm_mat_subsample_filt, empty_binary_sm_cols], axis=1)[propNet_nodes]
        return sm_mat_subsample_full
    else:
        return sm_mat_subsample_filt

# Function to quantile normalize a pandas DataFrame
# Code taken from: https://github.com/ShawnLYU/Quantile_Normalize/blob/master/quantile_norm.py
# Using implementation described on Wikipedia: https://en.wikipedia.org/wiki/Quantile_normalization
# data: Pandas DataFrame (propagated genotypes) where rows are samples (samples), and columns are features (genes)
# Returns df_out: Quantile normalized Pandas DataFrame with same orientation as data df
def qnorm(data):
    df = data.T
    df_out = df.copy(deep=True)
    dic = {}

    # Sort each patient by gene propagation value
    for col in df:
        dic.update({col:sorted(df[col])})
    sorted_df = pd.DataFrame(dic)

    # Rank averages for each gene across samples
    ranked_avgs = sorted_df.mean(axis = 1).tolist()

    # Construct quantile normalized Pandas DataFrame by assigning ranked averages to ranks of each gene for each sample
    for col in df_out:
        t = stats.rankdata(df[col]).astype(int)
        df_out[col] = [ranked_avgs[i-1] for i in t]
    return df_out.T

# Adapted from Matan Hofree's Matlab code in NBS
# data = features-by-samples propagated (or not) mutation profiles
# KNN_glap = Graph laplacian of regularization network
# Note: Make sure that the rows of Y are aligned correctly with the rows and columns of KNN_glap before passing into function
# k = Number of clusters for factorization
# H_init = Optional initial H array (k-by-samples)
# W_init = Optional nitial W array (features-by-k)
# gamma = Network regularization term constant
# eps = Small number precision
# update_gamma = Whether or not to update gamma regularization term on the fly
# gamma_factor = Factor to scale new gamma regularization term by (based on residual value ratios)
# Loop break conditions:
#   err_tol = Maximum value of reconstruction error function allowed for break
#   err_delta_tol = Maximum change in reconstruction error allowed for break
#   niter = Maximum number of iterations to execute before break
# verbose = print statements on update progress
# debug_mode = Returns intermediate values during updating if desired
def mixed_netNMF(data, KNN_glap, k, W_init=None, H_init=None, 
                 gamma=1000, update_gamma=True, gamma_factor=1, 
                 niter=250, eps=1e-15, err_tol=1e-4, err_delta_tol=1e-4, 
                 verbose=True, debug_mode=False):
    # Initialize H and W Matrices from data array if not given
    r, c = data.shape[0], data.shape[1]
    # Initialize H
    if H_init is None:
        H_init = np.random.rand(k,c)
        H = np.maximum(H_init, eps)
    else:
        H = np.copy(H_init)
    # Initialize W
    if W_init is None:
        W_init = np.linalg.lstsq(H.T, data.T)[0].T
        W_init = np.dot(W_init, np.diag(1/sum(W_init)))
        W = np.maximum(W_init, eps)
    else:
        W = np.copy(W_init)
    print 'W and H matrices initialized'
    # Get graph matrices from laplacian array
    D = np.diag(np.diag(KNN_glap)).astype(float)
    A = (D-KNN_glap).astype(float)
    print 'D and A matrices calculated'
    
    # Set mixed netNMF reporting variables
    optGammaIterMin, optGammaIterMax = 0, niter/2
    if debug_mode:
        resVal, resVal_Kreg, fitResVect, fitGamma, Wlist, Hlist = [], [], [], [], [], []
    XfitPrevious = np.inf

    # Updating W and H
    for i in range(niter):
        KWmat = np.dot(KNN_glap, W)
        Kres = np.trace(np.dot(W.T, KWmat)) # Un-scaled regularization term (originally sqrt of this value is taken)
        XfitThis = np.dot(W, H)
        WHres = np.linalg.norm(data-XfitThis) # Reconstruction error

        # Change in reconstruction error
        if i == 0:
            fitRes = np.linalg.norm(XfitPrevious)
        else:
            fitRes = np.linalg.norm(XfitPrevious-XfitThis)
        XfitPrevious = XfitThis
        # Tracking reconstruction errors and residuals
        if debug_mode:
            resVal.append(WHres)
            resVal_Kreg.append(Kres)
            fitResVect.append(fitRes)
            fitGamma.append(gamma)
            Wlist.append(W)
            Hlist.append(H)
        if (verbose) & (i%10==0):
            print 'Iteration >>', i, 'Mat-res:', WHres, 'K-res:', np.sqrt(Kres), 'Sum:', WHres+np.sqrt(Kres), 'Gamma:', gamma, 'Wfrob:', np.linalg.norm(W)
        if (err_delta_tol > fitRes) | (err_tol > WHres) | (i+1 == niter):
            if verbose:
                print 'NMF completed!'
                print 'Total iterations:', i+1
                print 'Final Reconstruction Error:', WHres
                print 'Final Reconstruction Error Delta:', fitRes
            numIter = i+1
            finalResidual = WHres
            break

        # Update Gamma
        if (update_gamma==True) & (gamma_factor!=0) & (i+1 <= optGammaIterMax) & (i+1 > optGammaIterMin):
            new_gamma = round((WHres/np.sqrt(Kres))*gamma_factor)
            gamma = new_gamma
        # Terms to be scaled by gamma
        KWmat_D = np.dot(D,W) 
        KWmat_W = np.dot(A,W)
            
        # Update W with network constraint
        W = W*((np.dot(data, H.T) + gamma*KWmat_W + eps) / (np.dot(W,np.dot(H,H.T)) + gamma*KWmat_D + eps))
        W = np.maximum(W, eps)
        W = W/matlib.repmat(np.maximum(sum(W),eps),len(W),1);        
        # Update H
        H = np.array([nnls(W, data[:,j])[0] for j in range(data.shape[1])]).T 
        # ^ Matan uses a custom fast non-negative least squares solver here, we will use scipy's implementation here
        H=np.maximum(H,eps)
    
    if debug_mode:
        return W, H, numIter, finalResidual, resVal, resVal_Kreg, fitResVect, fitGamma, Wlist, Hlist
    else:
        return W, H, numIter, finalResidual  

############################################################
# ---------- NBS Consensus Clustering Functions ---------- #
############################################################
import os
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hclust

# Constructs Hlist object for consensus clustering functions if NBS iterations were run in parallel and outputs saved to a folder
def Hlist_constructor_from_folder(folder, ext='.csv', normalize_H=False, verbose=False):
    co_clustering_results = [folder+fn for fn in os.listdir(folder) if fn.endswith(ext)]
    # Generate list of patient clusterings from netNMF
    Hlist = [pd.read_csv(fn, index_col=0) for fn in co_clustering_results]
    # Normalize H matrices if needed (to make columns comparable if not already done in decomposition)
    if normalize_H:
        Hlist_norm = []
        for H in Hlist:
            H_norm = np.dot(H,np.diag(1/H.sum()))
            Hlist_norm.append(pd.DataFrame(H_norm, index=H.index))
        if verbose:
            print 'Hlist constructed and normalized'
        return Hlist_norm
    else:
        if verbose:
            print 'Hlist constructed'
        return Hlist

# Takes a list of 'H' (patient-by-k) dataframes and performs 'hard' consensus clustering
# Using hierarchical clustering and average linkage
# Returns similarity table (distance is 1-similarity) and linkage map of patients
# Also returns cluster assignment map of patients if wanted
def consensus_hclust_hard(Hlist, k, assign_cluster=False, verbose=False):
    # Generate patient list
    pat_list = set()
    for H in Hlist:
        pat_list = pat_list.union(set(H.index))
    pat_list = sorted(list(pat_list))
    if verbose:
        print 'Constructing Hlist:', len(Hlist), 'cluster matrices', len(pat_list), 'samples'

    # Initialzie co-clustering tables
    co_clust_table = pd.DataFrame(0, index=pat_list, columns=pat_list)
    cluster_count = pd.DataFrame(0, index=pat_list, columns=pat_list)

    # Calculate patient similarities and linkage
    for H in Hlist:
        H.columns = range(1,len(H.columns)+1)
        # Update patient cluster count
        cluster_count.ix[H.index, H.index]+=1
        # Get cluster assignment for each patient
        cluster_assign = {i:[] for i in H.columns}
        for pat in H.index:
            cluster_assign[np.argmax(H.ix[pat])].append(pat)
        # Update co-clustering matrix with each cluster assignment
        for cluster in cluster_assign:
            cluster_pats = cluster_assign[cluster]
            co_clust_table.ix[cluster_pats, cluster_pats]+=1
    cc_hard_sim_table = co_clust_table.astype(float).divide(cluster_count.astype(float)).fillna(0)
    cc_hard_dist_table = 1-cc_hard_sim_table
    Z = hclust.linkage(dist.squareform(np.array(cc_hard_dist_table)), method='average')
    if assign_cluster:
        cluster_map = hclust.fcluster(Z, k, criterion='maxclust')
        cluster_assign = pd.Series({cc_hard_dist_table.index[i]:cluster_map[i] for i in range(len(cc_hard_dist_table.index))}, name='CC Hard, k='+repr(k))
        if verbose:
            print 'Hlist consensus constructed and sample clusters assigned'
        return cc_hard_sim_table, Z, cluster_assign
    else:
        if verbose:
            print 'Hlist consensus constructed and sample clusters assigned'
        return cc_hard_sim_table, Z

###############################################
# ---------- NBS Wrapper Functions ---------- #
###############################################
from network_evaluation_tools import data_import_tools as dit
from network_evaluation_tools import network_propagation as prop

# Wrapper function to run a single instance of network-regularized NMF on given somatic mutation data and network
# sm_mat = binary mutation matrix of data to perform network NMF on
# options = dictionary of options to set for various parts of netNMF execuction
# propNet = NetworkX graph object to propagate binary mutations over
# regNet_glap = Pandas DataFrame graph laplacian of network from influence matrix of propNet
def NBS_single(sm_mat, options, propNet=None, regNet_glap=None, verbose=True, save_path=None):
    # Set default NBS netNMF options
    NBS_options = {'pats_subsample_p':0.8,
                   'gene_subsample_p':0.8,
                   'min_muts':10,
                   'prop_data':True,
                   'prop_alpha':0.7,
                   'prop_symmetric_norm':False,
                   'qnorm_data':True,
                   'netNMF_k':4,
                   'netNMF_gamma':200,
                   'netNMF_update_gamma':False,
                   'netNMF_gamma_factor':1,
                   'netNMF_niter':250,
                   'netNMF_eps':1e-15,
                   'netNMF_err_tol':1e-4,
                   'netNMF_err_delta_tol':1e-4}
    # Update NBS netNMF options
    for option in options:
        NBS_options[option] = options[option]
    if verbose:
        print 'NBS options set:'
        for option in NBS_options:
            print '\t', option+':', NBS_options[option]
    
    # Check for correct input data
    if NBS_options['prop_data']:
        if type(propNet)!=nx.Graph:
            raise TypeError('Networkx graph object required for propNet')
    if (NBS_options['netNMF_gamma']!=0):
        if type(regNet_glap)!=pd.DataFrame:
            raise TypeError('netNMF regularization network laplacian (regNet_glap) must be given as Pandas DataFrame')
    # Subsample Data
    sm_mat_subsample = subsample_sm_mat(sm_mat, propNet=propNet, 
                                        pats_subsample_p=NBS_options['pats_subsample_p'], 
                                        gene_subsample_p=NBS_options['gene_subsample_p'], 
                                        min_muts=NBS_options['min_muts'])
    if verbose:
        print 'Somatic mutation data sub-sampling complete'
    # Propagate Data
    if NBS_options['prop_data']:
        prop_sm_data = prop.closed_form_network_propagation(propNet, sm_mat_subsample, symmetric_norm=NBS_options['prop_symmetric_norm'], alpha=NBS_options['prop_alpha'], verbose=verbose)
        if verbose:
            print 'Somatic mutation data propagated'
    else:
        prop_sm_data = sm_mat_subsample
        print 'Somatic mutation data not propagated'
    # Quantile Normalize Data
    if NBS_options['qnorm_data']:
        prop_data_qnorm = qnorm(prop_sm_data)
        if verbose:
            print 'Somatic mutation data quantile normalized'
    else:
        prop_data_qnorm = prop_sm_data
        print 'Somatic mutation data not quantile normalized'
    # Prepare data for mixed netNMF function
    propNet_nodes = propNet.nodes()
    data_arr = np.array(prop_data_qnorm.T.ix[propNet_nodes])
    regNet_glap_arr = np.array(regNet_glap.ix[propNet_nodes][propNet_nodes])
    # Mixed netNMF Result
    W, H, numIter, finalResid = mixed_netNMF(data_arr, regNet_glap_arr, NBS_options['netNMF_k'], W_init=None, H_init=None, 
                                             gamma=NBS_options['netNMF_gamma'], update_gamma=NBS_options['netNMF_update_gamma'], 
                                             gamma_factor=NBS_options['netNMF_gamma_factor'], niter=NBS_options['netNMF_niter'], 
                                             eps=NBS_options['netNMF_eps'], err_tol=NBS_options['netNMF_err_tol'], 
                                             err_delta_tol=NBS_options['netNMF_err_delta_tol'], verbose=verbose, debug_mode=False)
    # Save netNMF Result
    H_df = pd.DataFrame(H.T, index=prop_data_qnorm.index)
    if save_path is None:
        return H_df
    else:
        H_df.to_csv(save_path)
        if verbose:
            print 'netNMF result saved:', save_path
        return H_df

# Wrapper function to run a multiple instances of network-regularized NMF on given somatic mutation data and network
# and then perform consensus clustering on the result
# knnGlap_file must be a .csv file
def NBS_cc(sm_mat_file, network_file, config_file=None, calculate_knnGlap=True, knnGlap_file=None, verbose=True):
    # Load NBS options (if any given)
    if config_file is not None:
        f = open(config_file)
        config_lines = f.read().splitlines()
        NBS_options = {line.split('\t')[0]:line.split('\t')[1] for line in config_lines if not line.startswith('#')}
    else:
        NBS_options = {}
    if verbose:
        print 'User set NBS options:'
        for option in NBS_options:
            print option+':', NBS_options[option]
    # Load somatic mutation data
    sm_mat = dit.load_binary_mutation_data(sm_mat_file, filetype=NBS_options['sm_mat_filetype'], delimiter=NBS_options['sm_mat_delim'], verbose=True)
    # Load network
    network = dit.load_network_file(network_file)
    # Get knnGlap
    if calculate_knnGlap:
        knnGlap = network_inf_KNN_glap(network, gamma=NBS_options['knnGlap_gamma'], kn=NBS_options['knnGlap_kn'], verbose=verbose, save_path=NBS_options['knnGlap_save_path'])
    else:
        if knnGlap_file is None:
            raise ValueError('knnGlap file required')        
        else:
            knnGlap = pd.read_csv(knnGlap_file)
    # netNMF decomposition
    Hlist = []
    for i in range(NBS_options['niter']):
        Hlist.append(NBS_single(sm_mat, NBS_options, propNet=network, regNet_glap=knnGlap, verbose=False, save_path=None))
        if verbose:
            print 'NBS iteration:', i+1, 'complete'
    # Consensus Clustering
    NBS_cc_table, NBS_cc_linkage, NBS_cluster_assign = consensus_hclust_hard(Hlist, k, assign_cluster=True)
    if verbose:
        print 'Consensus Clustering complete'
    # Plot Consensus Cluster Map
    NBS_cluster_assign_cmap = cluster_color_assign(NBS_cluster_assign, name='Cluster Assignment')
    plot_cc_map(NBS_cc_table, NBS_cc_linkage, title=NBS_options['cc_map_title'], row_color_map=None, col_color_map=NBS_cluster_assign_cmap, save_path=NBS_options['cc_map_save_path'])
    if verbose:
        print 'Consensus Clustering map saved'        

################################################
# ---------- NBS Plotting Functions ---------- #
################################################
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test as multiv_lr_test

# Helper function for turning cluster assignments into color mappings (used for consensus clustering map figures)
def cluster_color_assign(cluster_assignments, name=None):
    k = max(cluster_assignments.value_counts().index)
    colors = sns.color_palette('hls', k)
    cluster_cmap = {i:colors[i-1] for i in range(1, k+1)}
    pat_colors = {}
    for pat in cluster_assignments.index:
        pat_colors[pat] = cluster_cmap[cluster_assignments.ix[pat]]
    return pd.Series(pat_colors, name=name)

# Function for plotting consensus clustering map
# Needs both the consensus clustering similarity table and linkage map from assignment
# Actual cluster assignments on col_color_map
# Cluster assignments to be compared passed to row_color_map
# If there are multiple mappings for row_color_map, it can be passed as a dataframe with the index space of the cc_table
def plot_cc_map(cc_table, linkage, title=None, row_color_map=None, col_color_map=None, save_path=None):
    plt.figure(figsize=(20,20))
    cg = sns.clustermap(cc_table, row_linkage=linkage, col_linkage=linkage, 
                        cmap='Blues', cbar_kws={'label': 'Co-Clustering'},
                        row_colors=row_color_map, col_colors=col_color_map, 
                        **{'xticklabels':'False', 'yticklabels':'False'})
    cg.cax.set_position([0.92, .12, .03, .59])
    cg.ax_heatmap.set_xlabel('')
    cg.ax_heatmap.set_xticks([])
    cg.ax_heatmap.set_ylabel('')
    cg.ax_heatmap.set_yticks([])
    cg.ax_row_dendrogram.set_visible(False)
    plt.suptitle(title, fontsize=20, x=0.6, y=0.95)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        return

# Function for plotting Kaplan Meier plot of cluster survivals
# Requires lifelines package
# clin_data_fn is the the clinical data of TCGA cohort from Broad Firehose
# cluster_assign ias a pandas Series of the patient cluster assignments from NBS with patient ID's as the index
# tmax is the maximum plot duration for the KMplot, but the logrank test always calculates to longest survival point
def cluster_KMplot(cluster_assign, clin_data_fn, title=None, lr_test=True, tmax=None, save_path=None):
    # Initialize KM plotter
    kmf = KaplanMeierFitter()
    # Load and format clinical data
    clin_data = pd.read_csv(clin_data_fn, sep='\t', index_col=0)
    clin_data = clin_data.T
    clin_data.index = [index.upper() for index in clin_data.index]
    surv = clin_data[['vital_status', 'days_to_death', 'days_to_last_followup']].fillna(0)
    surv['overall_survival'] = surv[['days_to_death', 'days_to_last_followup']].max(axis=1).map(lambda x: int(x))
    # Number of clusters
    clusters = sorted(list(cluster_assign.value_counts().index))
    k = len(clusters)
    # Initialize KM Plot Settings
    fig = plt.figure(figsize=(10, 7)) 
    ax = plt.subplot(1,1,1)
    colors = sns.color_palette('hls', k)
    cluster_cmap = {clusters[i]:colors[i] for i in range(k)}
    # Plot each cluster onto KM Plot
    for clust in clusters:
        clust_pats = cluster_assign[cluster_assign==clust].index.values
        clust_surv_data = surv.ix[clust_pats].dropna()
        kmf.fit(clust_surv_data.overall_survival, clust_surv_data.vital_status, label='Group '+str(clust)+' (n=' +  str(len(clust_surv_data)) + ')')
        kmf.plot(ax=ax, color=cluster_cmap[clust], ci_show=False)
    # Set KM plot limits and labels
    if tmax is not None:
        plt.xlim((0,tmax))
    plt.xlabel('Time (Days)', fontsize=16)
    plt.ylabel('Survival Probability', fontsize=16)
    # Multivariate logrank test
    if lr_test:
        cluster_survivals = pd.concat([surv, cluster_assign], axis=1).dropna().astype(int)
        p = multiv_lr_test(np.array(cluster_survivals.overall_survival), 
                           np.array(cluster_survivals[cluster_assign.name]), 
                           event_observed=np.array(cluster_survivals.vital_status)).p_value
        print 'Multi-Class Log-Rank P:', p
        plt.title(title+'\np='+repr(round(p, 6)), fontsize=24, y=1.02)
    else:
        plt.title(title, fontsize=24, y=1.02)
    # Save KM plot
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        return
