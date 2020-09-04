#!/usr/bin/env bash

# This script needs to be run from the deepFPlearn directory.
# Importantly, the conda environment needs to be set up and actived! For certain machines/HPC,
# we have a batch-job that does exactly that and then calls this file

python -m dfpl convert -f "data"
python -m dfpl train -f "validation/case_00/train_AC_S.json"
python -m dfpl train -f "validation/case_00/train_AC_X.json"
python -m dfpl train -f "validation/case_00/train_AC_D.json"

python -m dfpl train -f "validation/case_01/train.json"
python -m dfpl train -f "validation/case_02/train.json"
python -m dfpl train -f "validation/case_03/train.json"
python -m dfpl train -f "validation/case_04/train.json"
python -m dfpl train -f "validation/case_05/train.json"
python -m dfpl train -f "validation/case_06/train.json"

python -m dfpl predict -f "validation/case_07/predict.json"

python -m dfpl train -f "validation/case_11/train.json"
python -m dfpl train -f "validation/case_12/train.json"
python -m dfpl train -f "validation/case_13/train.json"
python -m dfpl train -f "validation/case_14/train.json"
python -m dfpl train -f "validation/case_15/train.json"
python -m dfpl train -f "validation/case_16/train.json"

python -m dfpl train -f "validation/case_31/train.json"
python -m dfpl train -f "validation/case_32/train.json"
python -m dfpl train -f "validation/case_33/train.json"