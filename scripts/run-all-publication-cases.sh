#!/usr/bin/env bash

# This script needs to be run from the deepFPlearn directory.
# Importantly, the conda environment needs to be set up and actived! For certain machines/HPC,
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
call_train "validation/case_02/train.json"

call_train "validation/case_03/train.json"

call_predict "validation/case_07/predict_bestAR03.json"
call_predict "validation/case_07/predict_bestED03.json"
call_predict "validation/case_07/predict_bestER03.json"
call_predict "validation/case_07/predict_fullER03.json"

call_train "validation/case_01/train_0p5.json"
call_train "validation/case_01/train_0p6.json"
call_train "validation/case_01/train_0p7.json"
call_train "validation/case_01/train_0p8.json"
call_train "validation/case_01/train_0p9.json"
call_train "validation/case_01/train_1p0.json"

call_train "validation/case_02/train_0p5.json"
call_train "validation/case_02/train_0p6.json"
call_train "validation/case_02/train_0p7.json"
call_train "validation/case_02/train_0p8.json"
call_train "validation/case_02/train_0p9.json"
call_train "validation/case_02/train_1p0.json"
