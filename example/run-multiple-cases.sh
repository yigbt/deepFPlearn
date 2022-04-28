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

call_train "train.json"

call_predict "predict.json"
