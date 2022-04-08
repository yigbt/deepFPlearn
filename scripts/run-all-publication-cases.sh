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

call_train "deepFPlearn_TRAINING/case_00/train_AE_S.json"
call_train "deepFPlearn_TRAINING/case_00/train_AE_D.json"
call_train "deepFPlearn_TRAINING/case_00/train_AE_Sider.json"
call_train "deepFPlearn_TRAINING/case_00/train_AE_Tox21.json"

call_train "deepFPlearn_TRAINING/case_01/train.json"

call_train "deepFPlearn_TRAINING/case_02/train.json"

call_train "deepFPlearn_TRAINING/case_03/train.json"

call_predict "deepFPlearn_TRAINING/case_04/predict_D-AR.json"
call_predict "deepFPlearn_TRAINING/case_04/predict_D-ED.json"
call_predict "deepFPlearn_TRAINING/case_04/predict_D-ER.json"

call_train "deepFPlearn_TRAINING/case_sider/train_SIDER-AE_D.json"
call_train "deepFPlearn_TRAINING/case_sider/train_SIDER-AE_Sider.json"

call_train "deepFPlearn_TRAINING/case_tox21/train_Tox21-AE_D.json"
call_train "deepFPlearn_TRAINING/case_tox21/train_Tox21-AE_Tox21.json"
