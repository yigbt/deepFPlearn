#!/bin/bash

#ml purge
#ml anaconda/5/5.0.1
#source activate rdkit2019

for M in /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/2019-10-16_311681247_1000/model*.h5;
do
    G=`basename $M`;
    H=${G//model\./};
    T=${H//\.h5/};

    echo $T;

    # files
    I="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/multiAOPtox.smiles.csv"
    O="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/results/multiAOPtox.smiles.$T.predictions.csv"

    echo $O;

    echo "python deepFPlearn-Predict.py -i $I -m $M -o $O -t smiles -k topological 1>runAll.out 2>runAll.err;"

#    head $O;

done
