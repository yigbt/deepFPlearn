#!/usr/bin/env bash

script_dir=$(dirname "$0")
D="$script_dir/data/"

# check if conda env exists, create if not, activate it
env=$(conda env list | grep 'dfpl_env' | wc -l)
if [[ $env -ne 1 ]]; then
  conda env create -f "$script_dir/../singularity_container/environment.yml"
fi
source activate dfpl_env

if [ -d $D ]; then
  export PYTHONPATH="$script_dir/.."
  python -m dfpl convert -f $D
fi
