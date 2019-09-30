#!/bin/bash

for M in /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/2019-09-30_16\:01\:46/model.AR.h5 #/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/2019-09-30_16\:01\:46/model.h5;
do
    G=`basename $M`;
    H=${G//model\./};
    T=${H//\.h5/};

    echo $T;

    # files
    I="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/all700000.chemicals.cid.smiles.csv"
    G=${I//\.csv/\.$T\.csv}
    O=${G//input/prediction}

    echo $O;

    echo "python deepFPlearn-Predict.py -i $I -m $M -o $O -t smile -k topological 1>runAll.out 2>runAll.err;";

    head $O;

done
