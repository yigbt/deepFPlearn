#!/usr/bin/env bash

# This script needs to be run from the deepFPlearn directory.
# Importantly, the conda environment needs to be set up and activated! For certain machines/HPC,
# we have a batch-job that does exactly that and then calls this file

function log_error() {
  echo "$@" 1>&2
}

function call_convert() {
  if [ -d "$1" ]; then
    python -m dfpl convert -f "$1"
  else
    log_error "Could not find directory for data conversion $1"
  fi
}

function call_train() {
  if [ -f "$1" ]; then
    python -m dfpl train -f "$1"
  else
    log_error "Could not find training file $1"
  fi
}

function call_predict() {
  if [ -f "$1" ]; then
    python -m dfpl predict -f "$1"
  else
    log_error "Could not find prediction file $1"
  fi
}

call_convert "data"

call_train "validation/case_00/train_AC_S.json"
call_train "validation/case_00/train_AC_D.json"

call_train "validation/case_01/train.json"
call_train "validation/case_01/train_0p5.json"
call_train "validation/case_01/train_0p6.json"
call_train "validation/case_01/train_0p7.json"
call_train "validation/case_01/train_0p8.json"
call_train "validation/case_01/train_0p9.json"
call_train "validation/case_01/train_1p0.json"

call_train "validation/case_02/train.json"
call_train "validation/case_02/train_0p5.json"
call_train "validation/case_02/train_0p6.json"
call_train "validation/case_02/train_0p7.json"
call_train "validation/case_02/train_0p8.json"
call_train "validation/case_02/train_0p9.json"
call_train "validation/case_02/train_1p0.json"

call_train "validation/case_03_S/train.json"
call_train "validation/case_03_Sext/train.json"

call_predict "validation/case_07_D/predict_bestAR03.json"
call_predict "validation/case_07_D/predict_bestED03.json"
call_predict "validation/case_07_D/predict_bestER03.json"
call_predict "validation/case_07_D/predict_fullAR03.json"
call_predict "validation/case_07_D/predict_fullED03.json"
call_predict "validation/case_07_D/predict_fullER03.json"

call_predict "validation/case_07_S/predict_bestAR03.json"
call_predict "validation/case_07_S/predict_bestED03.json"
call_predict "validation/case_07_S/predict_bestER03.json"
call_predict "validation/case_07_S/predict_fullAR03.json"
call_predict "validation/case_07_S/predict_fullED03.json"
call_predict "validation/case_07_S/predict_fullER03.json"

call_predict "validation/case_07_Sext/predict_bestAR03.json"
call_predict "validation/case_07_Sext/predict_bestED03.json"
call_predict "validation/case_07_Sext/predict_bestER03.json"
call_predict "validation/case_07_Sext/predict_fullAR03.json"
call_predict "validation/case_07_Sext/predict_fullED03.json"
call_predict "validation/case_07_Sext/predict_fullER03.json"

call_predict "validation/case_08_D/predict_bestARext03.json"
call_predict "validation/case_08_D/predict_bestEDext03.json"
call_predict "validation/case_08_D/predict_bestERext03.json"
call_predict "validation/case_08_D/predict_fullARext03.json"
call_predict "validation/case_08_D/predict_fullEDext03.json"
call_predict "validation/case_08_D/predict_fullERext03.json"

call_predict "validation/case_08_S/predict_bestARext03.json"
call_predict "validation/case_08_S/predict_bestEDext03.json"
call_predict "validation/case_08_S/predict_bestERext03.json"
call_predict "validation/case_08_S/predict_fullARext03.json"
call_predict "validation/case_08_S/predict_fullEDext03.json"
call_predict "validation/case_08_S/predict_fullERext03.json"

call_predict "validation/case_08_Sext/predict_bestARext03.json"
call_predict "validation/case_08_Sext/predict_bestEDext03.json"
call_predict "validation/case_08_Sext/predict_bestERext03.json"
call_predict "validation/case_08_Sext/predict_fullARext03.json"
call_predict "validation/case_08_Sext/predict_fullEDext03.json"
call_predict "validation/case_08_Sext/predict_fullERext03.json"
