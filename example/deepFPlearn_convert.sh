#!/bin/bash

D="example/data/"

# check if conda env exists, create if not, activate it
env=$(conda env list | grep 'dfpl_env' | wc -l)
if [[ $env -ne 1 ]]; then
  conda env create -f ../singularity_container/environment.yml
fi
source activate dfpl_env
cd ..

if [ -d $D ]; then
  python -m dfpl convert -f $D
fi
