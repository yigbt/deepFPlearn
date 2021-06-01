#!/bin/bash

F="example/train.json"

# check if conda env exists, create if not, activate it
env=$(conda env list | grep 'dfpl_env' | wc -l)
if [[ $env -ne 1 ]]; then
  conda env create -f ../singularity_container/environment.yml
fi
source activate dfpl_env
cd ..

# train the models as described in the .json file
if [ -f $F ]; then
  python -m dfpl train -f $F
fi

conda deactivate
