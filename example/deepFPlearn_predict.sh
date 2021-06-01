#!/usr/bin/env bash

script_dir=$(dirname "$0")
F="$script_dir/predict.json"

# check if conda env exists, create if not, activate it
env=$(conda env list | grep 'dfpl_env' | wc -l)
if [[ $env -ne 1 ]]; then
  conda env create -f "$script_dir/../singularity_container/environment.yml"
fi
source activate dfpl_env

# train the models as described in the .json file
if [ -f $F ]; then
  export PYTHONPATH="$script_dir/.."
  python -m dfpl predict -f $F
fi

conda deactivate
