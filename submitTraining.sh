#!/bin/bash

ml purge
ml anaconda/5/5.0.1
conda activate rdkit2019

d=`date +%Y-%m-%d_%T`;

INPUT="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.csv"
OUTDIR="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/${d}/"
EPOCHS=5000
log="${OUTDIR}train.log"
err="${OUTDIR}train.err"

mkdir -p $OUTDIR

python /home/hertelj/git-hertelj/code/deepFPlearn/deepFPlearn-Train.py -i $INPUT -o $OUTDIR  -t smiles -k topological -e $EPOCHS 1>$log 2>$err
