#!/usr/bin/env bash

source /home/patrick/build/local/miniconda3/etc/profile.d/conda.sh
conda activate rdkit2019
conda develop dfpl

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
