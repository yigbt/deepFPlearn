#!/usr/bin/env bash

source /home/patrick/build/local/miniconda3/etc/profile.d/conda.sh
conda activate rdkit2019
conda develop dfpl

bash scripts/run-all-cases.sh
