# Network Evaluation Tools

Network Evaluation Tools is a Python 2.7 package with corresponding examples for evaluating a network's ability to group a given node set in network proximity. This package was developed as a part of the work done in [citation](link). 

## Modules in this package
  - data_import_tools - 
  - gene_conversion_tools - 
  - miscellaneous_functions - 
  - network_evaluation_functions - 
  - network_propagation - 

## Version and Dendencies
Currently, pyNBS requires Python 2.7 - Python 2.7.13. Note that pyNBS may not work with Python 3.0+.
pyNBS requires: 
  - Argparse >= 1.1
  - NetworkX >= 1.11
  - Numba >= 0.32.0
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
The following network analyses can be performed either from a Jupyter Notebook (see ```Network Evaluation Notebooks``` folder) or from the command line (see ```Network Evaluation Scripts``` folder). Jupyter notebooks are documented within the notebook and the documentation for the python scripts can be seen using the command ```python [script_name].py -h```. <br>
    1. Fast Network Evaluation  
        * Jupyter Notebook -
        * From the command line -  
    2. Full Network Evaluation  
        * Jupyter Notebook -
        * From the command line -

## Data provided in this repository (see ```Data``` Folder)
 - _DisGeNET / Oncogenic Component Gene Sets_ -
 - _Network performance (AUPRCs) on DisGeNET / Oncogenic Component Gene Sets_ -
 - _Network performance effect sizes on DisGeNET / Oncogenic Component Gene Sets_ -      

## Issues
Please feel free to post issues/bug reports. Questions can be sent to jkh013@ucsd.edu



