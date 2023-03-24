import math
import networkx as nx

import pytest
from network_evaluation_tools import data_import_tools as dit
from network_evaluation_tools import network_evaluation_functions as nef
from network_evaluation_tools import network_propagation as prop
import pandas as pd
import numpy as np
import pickle

network_test_file = '../Data/Networks/YoungvsOld_UP.csv'
disease_test_file = '../Data/Evaluations/DisGeNET_genesets.txt'
networkx_test_file = '../Data/NetworkCYJS/graph1_Young_Old_Fuzzy_95.pkl'

AUPRC_values = {'Carcinoma, Lewis Lung': 0.5136054421768708, 'Fanconi Anemia': 0.5048184241212726,
                'Endometrial adenocarcinoma': 0.5036461554318696, 'Follicular adenoma': -1.0,
                'Intracranial Aneurysm': -1.0}
network = dit.load_network_file('../Data/Networks/YoungvsOld_UP.csv', delimiter=',', verbose=True)
genesets = dit.load_node_sets('../Data/Evaluations/DisGeNET_genesets.txt')
genesets = {'Carcinoma, Lewis Lung': genesets['Carcinoma, Lewis Lung'],
            'Fanconi Anemia': genesets['Fanconi Anemia'],
            'Endometrial adenocarcinoma': genesets['Endometrial adenocarcinoma'],
            'Follicular adenoma': genesets['Follicular adenoma'],
            'Intracranial Aneurysm': genesets['Intracranial Aneurysm'],
            'Muscle Weakness': genesets['Muscle Weakness']
            }
genesets_p = {'Carcinoma, Lewis Lung': 0.5921,
              'Fanconi Anemia': 0.5589,
              'Endometrial adenocarcinoma': 0.5921,
              'Follicular adenoma': 0.649,
              'Intracranial Aneurysm': float('inf'),
              'Muscle Weakness': float('inf')}
alpha = 0.684


def test_construct_prop_kernel():
    """
    This test generates the kernel based on a specific network \
    of 206 nodes. If the network for example changes, make sure to
    edit the last 2 assertions on this test.

    :return:
    """
    _network = dit.load_network_file(network_test_file, delimiter=',', verbose=True)
    _gene_sets = dit.load_node_sets(disease_test_file)
    _gene_sets_p = nef.calculate_p(_network, _gene_sets)  # calculate the sub-sampling rate p for each node set
    _alpha = prop.calculate_alpha(_network)  # Calculate the Network Alpha
    kernel = nef.construct_prop_kernel(_network, alpha=_alpha, verbose=True)
    assert isinstance(kernel, pd.DataFrame)
    assert kernel.shape == (len(_network.nodes), len(_network.nodes))  # Propagate using the random walk model
    assert kernel['AIMP2']['AIMP2'] == 0.33562464152566673
    assert kernel['ARID5A']['ARL8B'] == 0
