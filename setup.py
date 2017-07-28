"""
Setup module adapted from setuptools code. See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

setup(
	name='network_evaluation_tools',
	version='0.0.2',
	description='Module to perform patient and molecular network evaluation as described in Huang, Carlin, et al.',
	url='https://github.com/pypa/sampleproject',
	author='Justin Huang',
	author_email='jkh013@ucsd.edu',
	license='MIT',
	classifiers=[
		'Development Status :: 2 - Pre-Alpha',
		'Intended Audience :: Science/Research',
		'Topic :: Software Development :: Build Tools',
		'License :: OSI Approved :: MIT License',
		'Programming Language :: Python :: 2.7'
	],
	packages=find_packages(exclude=['copy', 'itertools', 'os', 're', 'time']),
	install_requires=[
        'argparse>=1.1',
        'networkx>=1.11',
        'numba>=0.32.0',
        'numpy>=1.11.0',
        'matplotlib>=1.5.1',
        'multiprocessing>0.70',
        'pandas>=0.19.0',
        'requests>=2.13.0',
        'scipy>=0.17.0',
        'scikit-learn>=0.17.1',
        'seaborn>=0.7.1']
)