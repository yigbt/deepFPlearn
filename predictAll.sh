#!/bin/bash

for M in /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/model*.h5; 
do 
    G=`basename $M`; 
    H=${G//model\./}; 
    T=${H//\.h5/}; 

    echo $T; 

    # files
    I="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/multiAOPtox.smiles.csv"
    O=${I//\.csv/\.$T\.csv}

    echo $O;
    
    python deepFPlearn-Predict.py -i $I -m $M -o $O -t smile -k topological 1>runAll.out 2>runAll.err;

    head $O;

done 
