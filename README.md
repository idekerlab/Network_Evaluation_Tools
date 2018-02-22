# Network Evaluation Tools

Network Evaluation Tools is a Python 2.7 package with corresponding examples for evaluating a network's ability to group a given node set in network proximity. This package was developed as a part of the work done in [citation](link). 

## Modules in this package
  - _data_import_tools_ - This module contains functions for helping import network files and gene set files for analysis.
  - _gene_conversion_tools_ - This module contains functions for helping convert, filter, and save networks from their raw database form. Used in the Network Processing Jupyter Notebooks.
  - _miscellaneous_functions_ - This module contains various functions developed to help with analysis along the way. These functions are not well tested and may contain bugs. These functions were generally used to determine other network performance metrics on network recovery of gene sets.
  - _network_evaluation_functions_ - This module contains many of the core functions of the set-based network evaluation algorithm.
  - _network_propagation_ - This module contains functions to help with network propagation steps used in the set-based network evaluation algorithm.

## Version and Dendencies
Currently, the network_evaluation_tools package requires Python 2.7 - Python 2.7.13. Note that some functions in this package may not work with Python 3.0+.
network_evaluation_tools requires: 
  - Argparse >= 1.1
  - NetworkX >= 1.11
  - Numpy >= 1.11.0
  - Matplotlib >= 1.5.1
  - Pandas >= 0.19.0
  - Requests >= 2.13.0
  - Scipy >= 0.17.0
  - Scikit-learn >= 0.17.1

Note:
- In Pandas v0.20.0+, the ```.ix```indexer has been deprecated. There may be warning regarding this issue, yet the function still works.

## Installation
1. Clone the repository 
2. cd to new respository
3. Execute following command:  
```python setup.py install```

## Network analysis
1. If the network needs to be normalized to a particular naming scheme:<br>
A Jupyter Notebook describing how each network was processed from the raw download file in the original [paper](Link) can be found in the ```Network Processing Notebooks``` folder.<br>
2. There are two ways to perform the network evaluation on a gene set:<br>
The following network analyses can be performed either from a Jupyter Notebook or from the command line (see ```Network Evaluation Examples``` folder). Jupyter notebooks are documented within the notebook and the documentation for the python scripts can be seen using the command ```python [script_name].py -h```. <br>

## Data provided in this repository (see ```Data``` Folder)
 - Database Citations - An Excel file containing details about all of the networks used in the original paper's analysis and affiliated citations for all of the databases used.
 - _DisGeNET / Oncogenic Component Gene Sets_ - Two tab separated files, each line containing a gene set from either DisGeNET or the Oncogenic Component collection. The first column of each file is the name of the gene set followed by the list of genes associated with that given gene set on the same line.
 - _Network performance (AUPRCs) on DisGeNET / Oncogenic Component Gene Sets_ - Two csv files containing the raw Z-normalized AUPRC scores (network performance scores) of each network analyzed on each gene set analyzed from DisGeNET or the Oncogenic Component gene set collection.
 - _Network performance effect sizes on DisGeNET / Oncogenic Component Gene Sets_ - Two csv files containing the relative performance gain of each network's AUPRC score over the median null AUPRC score for each gene set analyzed from DisGeNET or the Oncogenic Component gene set collection.

## Issues
Please feel free to post issues/bug reports. Questions can be sent to jkh013@ucsd.edu

## License
See the [LICENSE](https://github.com/huangger/Network_Evaluation_Tools/blob/master/LICENSE.txt) file for license rights and limitations (MIT).


