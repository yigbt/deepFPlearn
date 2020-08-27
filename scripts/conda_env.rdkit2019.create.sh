#!/bin/bash

# I used the following command to create the conda environment rdkit2019
# Date: Sep 19, 2019
conda create -n rdkit2019 -c conda-forge -c bioconda -c r -c anaconda \
  python \
  pip \
  tensorflow \
  keras \
  numpy \
  pandas \
  rdkit \
  scipy \
  tensorboard \
  matplotlib \
  scikit-learn \
  seaborn markdown \
  ncurses \
  pcre \
  yaml \
  pyyaml \
  pytest \
  jsonpickle

# Export environment requirements
# The --from-history ensures that only packages go into the yaml file that we *initially* specified.
# This should make the script cross-platform since many dependencies that are also installed are
# system-dependent.
conda env export -n rdkit2019 --from-history > conda_env.rdkit2019.yml

# To create exactly this environment call
conda env create -f conda_env.rdkit2019.yml

# Activate the environment
source activate rdkit2019

# .. and start the deepFPlearn tool(s)

# .. or start (e.g.) pycharm and set this conda environment in the interpreter settings!
