#!/usr/bin/env bash

python -m dfpl convert -f "data"
python -m dfpl train -f "validation/case_00/train_AC_S.json"
python -m dfpl train -f "validation/case_00/train_AC_X.json"
python -m dfpl train -f "validation/case_00/train_AC_D.json"
