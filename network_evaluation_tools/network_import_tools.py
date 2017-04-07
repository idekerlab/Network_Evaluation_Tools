########################################################
# ---------- Network Directory Import Tools ---------- #
########################################################

import pandas as pd
import itertools
import networkx as nx
import time
import os

# Get networks in directory
def get_networks(wd, suffix, file_ext='.sif'):
    network_files = {}
    for f in os.listdir(wd):
        if f.endswith(file_ext) and f.split(file_ext)[0].endswith(suffix):
            network_files[f.split(suffix)[0]]=f
    return network_files

def load_networks(wd, network_file_map, delimiter='\t'):
    # Initialize dictionaries
    networks, network_edges, network_nodes = {}, {}, {}
    
    # Loading network and network properties
    for network_name in network_file_map:
        loadtime = time.time()
        # Load network edges (via sif)
        f = open(wd+network_file_map[network_name])
        lines = f.read().splitlines()
        edgelist = [tuple(line.split(delimiter)) for line in lines]
        network_edges[network_name] = edgelist
        # Construct list of network nodes from edge list
        network_edges[network_name] = edgelist
        nodes = list(set(itertools.chain.from_iterable(edgelist)))
        network_nodes[network_name] = nodes
        # Construct NetworkX object from edge list
        network = nx.Graph()
        network.add_edges_from(edgelist)
        networks[network_name]=network
        print network_name, 'network data loaded:', round(time.time()-loadtime, 2), 'seconds'
        
    # Return data structure
    return networks, network_edges, network_nodes