#!/bin/bash

# these are the calls that I started at EVE

input="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.fingerprints.csv"
outpath="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/"

# ER ###################################################################################

screen -S ERtuning # screen 0

# First I tune batch Size and Epochs at once

# --batchSizes: 32 128
# --epochs: 100 500 1000

ml purge
ml git
ml anaconda/5/5.0.1
source activate rdkit2019

target='ER'
Lfile="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E.32-128.100-500-1000."$target".log"
Efile=${Lfile//\.log/\.err}

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 32 128 --epochs 100 500 1000 1> $Lfile 2> $Efile &
#18987

# --> batchSize: 128
# --> Epochs: 100

# Now, let's tune the optimizer
LfileO="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O.128.100.diverse."$target".log"
EfileO=${Lfile//\.log/\.err}

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 128 --epochs 100 --optimizers 'SGD' 'RMSprop' 'Adagrad' 'Adadelta' 'Adam' 'Adamax' 'Nadam'  1> $LfileO 2> $EfileO &
#26272
#27942 BS 128, E 100

# and now, the activation functions
LfileA="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O-A.128.100.SGD.diverse."$target".log"
EfileA=${Lfile//\.log/\.err}

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 128 --epochs 500 --optimizers 'SGD' --activations 'sigmoid' 'relu'  1> $LfileA 2> $EfileA &
#26384

#[detached from 17318.ERtuning]

# reattach screen with
# screen -r 17318

# AR ###################################################################################

screen -S ARtuning

# First I tune batch Size and Epochs at once

# --batchSizes: 32 128
# --epochs: 100 500 1000

ml purge
ml git
ml anaconda/5/5.0.1
source activate rdkit2019

target='AR'
Lfile="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E.32-128.100-500-1000."$target".log"
Efile=${Lfile//\.log/\.err}

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 32 128 --epochs 100 500 1000 1> $Lfile 2> $Efile &
#16120

# --> batchSize: 128
# --> Epochs: 100

# Now, let's tune the optimizer
LfileO="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O.128.100.diverse."$target".log"
EfileO=${Lfile//\.log/\.err}

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 128 --epochs 100 --optimizers 'SGD' 'RMSprop' 'Adagrad' 'Adadelta' 'Adam' 'Adamax' 'Nadam'  1> $LfileO 2> $EfileO &
#16476
#30144 

# and now, the activation functions
LfileA="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O-A.128.500.SGD.diverse."$target".log"
EfileA=${Lfile//\.log/\.err}

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 128 --epochs 500 --optimizers 'SGD' --activations 'sigmoid' 'relu'  1> $LfileA 2> $EfileA &
#16601 BS 128, E 100

#[detached from 7065.ARtuning]

# reattach screen with
# screen -r 7065
