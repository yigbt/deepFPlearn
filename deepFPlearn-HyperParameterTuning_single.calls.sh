#!/bin/bash

# these are the calls that I started at EVE

input="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.fingerprints.csv"
outpath="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/"

# First I tune batch Size and Epochs at once

screen -S ERtuning # screen 0

# --batchSizes: 32 128
# --epochs: 100 500 1000

ml anaconda/5/5.0.1
source activate rdkit2019

target='ER'
Lfile="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E.32-128.100-500-1000."$target".log"
Efile=${Lfile//\.log/\.err}

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 32 128 --epochs 100 500 1000 1> $Lfile 2> $Efile &

# Now, let's tune the optimizer



# and now, the activation functions
