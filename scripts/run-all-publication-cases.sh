#!/usr/bin/env bash

# This script needs to be run from the deepFPlearn directory.
# Importantly, the conda environment needs to be set up and actived! For certain machines/HPC,
# we have a batch-job that does exactly that and then calls this file

python -m dfpl convert -f "data"
python -m dfpl train -f "validation/case_00/train_AC_S.json"
python -m dfpl train -f "validation/case_00/train_AC_D.json"

python -m dfpl train -f "validation/case_01/train.json"
python -m dfpl train -f "validation/case_01/train_0p5.json"
python -m dfpl train -f "validation/case_01/train_0p6.json"
python -m dfpl train -f "validation/case_01/train_0p7.json"
python -m dfpl train -f "validation/case_01/train_0p8.json"
python -m dfpl train -f "validation/case_01/train_0p9.json"
python -m dfpl train -f "validation/case_01/train_1p0.json"

python -m dfpl train -f "validation/case_02/train.json"
python -m dfpl train -f "validation/case_02/train_0p5.json"
python -m dfpl train -f "validation/case_02/train_0p6.json"
python -m dfpl train -f "validation/case_02/train_0p7.json"
python -m dfpl train -f "validation/case_02/train_0p8.json"
python -m dfpl train -f "validation/case_02/train_0p9.json"
python -m dfpl train -f "validation/case_02/train_1p0.json"

python -m dfpl train -f "validation/case_03/train.json"

python -m dfpl predict -f "validation/case_07/predict_bestER03.json"
python -m dfpl predict -f "validation/case_07/predict_bestARext03.json"
python -m dfpl predict -f "validation/case_07/predict_bestED03.json"
