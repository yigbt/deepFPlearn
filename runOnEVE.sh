#!/bin/bash

ml purge
ml anaconda/5/5.0.1

# if conda-env not already installed, create it:
#conda env create -f conda_env.rdkit2019.yaml

# otherwise load it
conda activate rdkit2019


### TRAINING

# use this to not overwrite existing trained models!
d=`date +%Y-%m-%d_%T`;

INPUT="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.csv"
OUTDIR="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/${d}/"
EPOCHS=200
log="${OUTDIR}train.log"
err="${OUTDIR}train.err"

mkdir -p $OUTDIR

python /home/hertelj/git-hertelj/code/deepFPlearn/deepFPlearn-Train.py -i $INPUT -o $OUTDIR  -t smiles -k topological -e $EPOCHS 1>$log 2>$err

### PREDICTIONS

INPUT="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/multiAOPtox.smiles.csv"
MODEL="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/model.AR.h5"
OUTPUT="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/prediction/multiAOPtox.smiles.predictions.AR.csv"

python /home/hertelj/git-hertelj/code/deepFPlearn/deepFPlearn-Predict.py -i $INPUT -m $MODEL -o $OUTPUT -t smiles -k topological
