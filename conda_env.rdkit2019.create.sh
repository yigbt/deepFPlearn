#!/bin/bash

# I used the following command to create the conda environment rdkit2019
# Date: Sep 19, 2019
conda create -n rdkit2019 -c conda-forge -c bioconda -c r -c anaconda python pip tensorflow keras numpy pandas rdkit scipy tensorboard matplotlib scikit-learn seaborn markdown ncurses pcre yaml pyyaml

# I exported this environment to this yaml configuration file wich
# then includes all installed package releases
conda env export -n rdkit2019 > conda_env.rdkit2019.yaml

# To create exactly this environment call
conda env create -f conda_env.rdkit2019.yml

# Activate the environment
source activate rdkit2019

# .. and start the deepFPlearn tool(s)

# .. or start (e.g.) pycharm and set this conda environment in the interpreter settings!
