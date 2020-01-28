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
#Lfile="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E.32-128.100-500-1000."$target".log"
Lfile="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E.32-64-128-256.10-50-100-500."$target".log"
Efile=${Lfile//\.log/\.err}

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 32 64 128 256 --epochs 10 50 100 500 1> $Lfile 2> $Efile &
#2240

# grep -P ',[12]' /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/Sun_etal_dataset.fingerprints.hpTuningResults.01-EpochsBatchSizeER.csv

#  9,50,128,0.8354197382926941,0.013692773675255714,2
# 12,10,256,0.8392203688621521,0.013122546508458705,1

#### now, lets tune optimizers

LfileO="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O.64.10.diverse."$target".log"
EfileO=${LfileO//\.log/\.err}

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 256 --epochs 10 --optimizers 'SGD' 'RMSprop' 'Adagrad' 'Adadelta' 'Adam' 'Adamax' 'Nadam'  1> $LfileO 2> $EfileO &
#19712

# grep -P ',[12]' /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/Sun_etal_dataset.fingerprints.hpTuningResults.02-OptimizerER.csv

# 4,  Adam,0.8343001723289489,0.010741602717115582,1
# 5,Adamax,0.8313934087753296,0.012607300681142583,2

#### now, lets tune activation functions

LfileA="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O-A.64.10.Adam.diverse."$target".log"
EfileA=${LfileA//\.log/\.err}
#[detached from 22945.ERtuning]

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 256 --epochs 10 --optimizers 'Adam' --activations 'sigmoid' 'relu'  1> $LfileA 2> $EfileA &
# 8149

## The tuning of epoch did not work out well.. I could achieve a lot
## better training results with mor epochs than the hp tuning resulted
## in!!
### So, I decided to fix the batch_size and epochs to 128 and 250, resp.

# tune optimizer again

LfileO="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O.128.250.diverse."$target".log"
EfileO=${LfileO//\.log/\.err}

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 128 --epochs 100 --optimizers 'SGD' 'RMSprop' 'Adagrad' 'Adadelta' 'Adam' 'Adamax' 'Nadam'  1> $LfileO 2> $EfileO &
# 318

# tune activation function again
LfileA="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O-A.128.100.Adam.diverse."$target".log"
EfileA=${LfileA//\.log/\.err}
python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 128 --epochs 100 --optimizers 'Adam' --activations 'sigmoid' 'relu'  1> $LfileA 2> $EfileA &
# 4800

# OTHER targets #######################################################################
screen -S ARtuning
ml purge
ml git
ml anaconda/5/5.0.1
source activate rdkit2019
cd git-hertelj/code/2019_deepFPlearn/
input="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.fingerprints.csv"
outpath="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/"
target='AR'
Lfile="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E.32-64-128-256.10-50-100-500."$target".log"
Efile=${Lfile//\.log/\.err}
python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 32 64 128 256 --epochs 10 50 100 500 1> $Lfile 2> $Efile &
#6837

# grep -P ',[12]' /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/Sun_etal_dataset.fingerprints.hpTuningResults.01-EpochsBatchSizeAR.csv

#  4, 10, 64,0.8557550191879273,0.006214979485192407,1
# 14,100,256,0.854065477848053, 0.006025498365485959,2

#### now, lets tune optimizers

LfileO="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O.64.10.diverse."$target".log"
EfileO=${LfileO//\.log/\.err}

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 64 --epochs 10 --optimizers 'SGD' 'RMSprop' 'Adagrad' 'Adadelta' 'Adam' 'Adamax' 'Nadam'  1> $LfileO 2> $EfileO &
#20582

# grep -P ',[12]' /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/Sun_etal_dataset.fingerprints.hpTuningResults.02-OptimizerAR.csv

# 4,  Adam,0.8456177473068237,0.015088132666807023,2
# 5,Adamax,0.8494192242622376,0.00814120900528407, 1

#### now, lets tune activation functions

LfileA="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O-A.64.10.Adamax.diverse."$target".log"
EfileA=${LfileA//\.log/\.err}
#[detached from 22945.ERtuning]

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 64 --epochs 10 --optimizers 'Adamax' --activations 'sigmoid' 'relu'  1> $LfileA 2> $EfileA &
# 10069

### So, I decided to fix the batch_size and epochs to 128 and 250, resp.

# tune optimizer again

LfileO="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O.128.250.diverse."$target".log"
EfileO=${LfileO//\.log/\.err}

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 128 --epochs 250 --optimizers 'SGD' 'RMSprop' 'Adagrad' 'Adadelta' 'Adam' 'Adamax' 'Nadam'  1> $LfileO 2> $EfileO &
# 1038

# tune activation function again
LfileA="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O-A.128.100.Adamax.diverse."$target".log"
EfileA=${LfileA//\.log/\.err}
python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 128 --epochs 100 --optimizers 'Adamax' --activations 'sigmoid' 'relu'  1> $LfileA 2> $EfileA &
# 5042


#[detached from 4878.ARtuning] --------------------------------------------------------
screen -S TRtuning
ml purge
ml git
ml anaconda/5/5.0.1
source activate rdkit2019
cd git-hertelj/code/2019_deepFPlearn/
input="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.fingerprints.csv"
outpath="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/"
target='TR'
Lfile="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E.32-64-128-256.10-50-100-500."$target".log"
Efile=${Lfile//\.log/\.err}
python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 32 64 128 256 --epochs 10 50 100 500 1> $Lfile 2> $Efile &
#8283
# grep -P ',[12]' /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/Sun_etal_dataset.fingerprints.hpTuningResults.01-EpochsBatchSizeTR.csv

#  0,10, 32,0.9294845342636109,0.00661496964224313,1
#  4,10, 64,0.9294845342636109,0.00661496964224313,1
# 12,10,256,0.9294845342636109,0.00661496964224313,1

#### now, lets tune optimizers

LfileO="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O.64.10.diverse."$target".log"
EfileO=${LfileO//\.log/\.err}

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 64 --epochs 10 --optimizers 'SGD' 'RMSprop' 'Adagrad' 'Adadelta' 'Adam' 'Adamax' 'Nadam'  1> $LfileO 2> $EfileO &
#20884

# grep -P ',[12]' /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/Sun_etal_dataset.fingerprints.hpTuningResults.02-OptimizerTR.csv

# 0,SGD,     0.9294845342636109,0.00661496964224313,1
# 1,RMSprop, 0.9294845342636109,0.00661496964224313,1
# 2,Adagrad, 0.9294845342636109,0.00661496964224313,1
# 3,Adadelta,0.9294845342636109,0.00661496964224313,1
# 4,Adam,    0.9294845342636109,0.00661496964224313,1
# 6,Nadam,   0.9294845342636109,0.00661496964224313,1

#### now, lets tune activation functions

LfileA="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O-A.64.10.Adam.diverse."$target".log"
EfileA=${LfileA//\.log/\.err}

### So, I decided to fix the batch_size and epochs to 128 and 250, resp.

# tune optimizer again

LfileO="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O.128.250.diverse."$target".log"
EfileO=${LfileO//\.log/\.err}

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 128 --epochs 250 --optimizers 'SGD' 'RMSprop' 'Adagrad' 'Adadelta' 'Adam' 'Adamax' 'Nadam'  1> $LfileO 2> $EfileO &
# 1753

# tune activation function again
LfileA="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O-A.128.100.Adam.diverse."$target".log"
EfileA=${LfileA//\.log/\.err}
python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 128 --epochs 100 --optimizers 'Adam' --activations 'sigmoid' 'relu'  1> $LfileA 2> $EfileA &
# 

#[detached from 22945.ERtuning]

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 64 --epochs 10 --optimizers 'Adam' --activations 'sigmoid' 'relu'  1> $LfileA 2> $EfileA &
# 11223

#[detached from 7508.TRtuning] --------------------------------------------------------

screen -S GRtuning
ml purge
ml git
ml anaconda/5/5.0.1
source activate rdkit2019
cd git-hertelj/code/2019_deepFPlearn/
input="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.fingerprints.csv"
outpath="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/"
target='GR'
Lfile="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E.32-64-128-256.10-50-100-500."$target".log"
Efile=${Lfile//\.log/\.err}
python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 32 64 128 256 --epochs 10 50 100 500 1> $Lfile 2> $Efile &
#10511
# grep -P ',[12]' /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/Sun_etal_dataset.fingerprints.hpTuningResults.01-EpochsBatchSizeGR.csv

#  0,10, 32,0.9188707351684571,0.008356397227460621,1
# 13,50,256,0.9184579372406005,0.008263388287393994,2

#### now, lets tune optimizers

LfileO="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O.32.10.diverse."$target".log"
EfileO=${LfileO//\.log/\.err}

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 32 --epochs 10 --optimizers 'SGD' 'RMSprop' 'Adagrad' 'Adadelta' 'Adam' 'Adamax' 'Nadam'  1> $LfileO 2> $EfileO &
#21616

# grep -P ',[12]' /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/Sun_etal_dataset.fingerprints.hpTuningResults.02-OptimizerGR.csv

# 1,RMSprop,0.9184560298919677,0.005339372029139733,1
# 6,  Nadam,0.9176321387290954,0.007490117200990504,2

#### now, lets tune activation functions

LfileA="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O-A.32.10.RMSprop.diverse."$target".log"
EfileA=${LfileA//\.log/\.err}

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 32 --epochs 10 --optimizers 'RMSprop' --activations 'sigmoid' 'relu'  1> $LfileA 2> $EfileA &
# 14509

### So, I decided to fix the batch_size and epochs to 128 and 250, resp.

# tune optimizer again

LfileO="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O.128.250.diverse."$target".log"
EfileO=${LfileO//\.log/\.err}

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 128 --epochs 250 --optimizers 'SGD' 'RMSprop' 'Adagrad' 'Adadelta' 'Adam' 'Adamax' 'Nadam'  1> $LfileO 2> $EfileO &
# 2354
# tune activation function again
LfileA="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O-A.128.100.RMSprop.diverse."$target".log"
EfileA=${LfileA//\.log/\.err}
python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 128 --epochs 100 --optimizers 'RMSprop' --activations 'sigmoid' 'relu'  1> $LfileA 2> $EfileA &
# 6823

#[detached from 9057.GRtuning] --------------------------------------------------------

screen -S PPARtuning
ml purge
ml git
ml anaconda/5/5.0.1
source activate rdkit2019
cd git-hertelj/code/2019_deepFPlearn/
input="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.fingerprints.csv"
outpath="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/"
target='PPARg'
Lfile="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E.32-64-128-256.10-50-100-500."$target".log"
Efile=${Lfile//\.log/\.err}
python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 32 64 128 256 --epochs 10 50 100 500 1> $Lfile 2> $Efile &
#12696
# grep -P ',[12]' /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/Sun_etal_dataset.fingerprints.hpTuningResults.01-EpochsBatchSizePPARg.csv

#  8,10,128,0.8971171259880066,0.013991627130374872,1
# 12,10,256,0.8923050880432128,0.013736195081539447,2

#### now, lets tune optimizers

LfileO="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O.128.10.diverse."$target".log"
EfileO=${LfileO//\.log/\.err}

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 128 --epochs 10 --optimizers 'SGD' 'RMSprop' 'Adagrad' 'Adadelta' 'Adam' 'Adamax' 'Nadam'  1> $LfileO 2> $EfileO &
#22412

# grep -P ',[12]' /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/Sun_etal_dataset.fingerprints.hpTuningResults.02-OptimizerPPARg.csv

# 0,    SGD,0.8936783313751221,0.013829235711519797,1
# 1,RMSprop,0.8936783313751221,0.013829235711519797,1
# 2,Adagrad,0.8936783313751221,0.013829235711519797,1
# 6,  Nadam,0.8936783313751221,0.013829235711519797,1

#### now, lets tune activation functions

LfileA="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O-A.64.10.SGD.diverse."$target".log"
EfileA=${LfileA//\.log/\.err}
#[detached from 22945.ERtuning]

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 64 --epochs 10 --optimizers 'SGD' --activations 'sigmoid' 'relu'  1> $LfileA 2> $EfileA &
# 13568

### So, I decided to fix the batch_size and epochs to 128 and 250, resp.

# tune optimizer again

LfileO="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O.128.250.diverse."$target".log"
EfileO=${LfileO//\.log/\.err}

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 128 --epochs 250 --optimizers 'SGD' 'RMSprop' 'Adagrad' 'Adadelta' 'Adam' 'Adamax' 'Nadam'  1> $LfileO 2> $EfileO &
# 2570
LfileA="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O-A.128.100.Adagrad.diverse."$target".log"
EfileA=${LfileA//\.log/\.err}
python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 128 --epochs 100 --optimizers 'Adagrad' --activations 'sigmoid' 'relu'  1> $LfileA 2> $EfileA &
# 7556


#[detached from 11294.PPARtuning] --------------------------------------------------------

screen -S AROtuning
ml purge
ml git
ml anaconda/5/5.0.1
source activate rdkit2019
cd git-hertelj/code/2019_deepFPlearn/
input="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.fingerprints.csv"
outpath="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/"
target='Aromatase'
Lfile="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E.32-64-128-256.10-50-100-500."$target".log"
Efile=${Lfile//\.log/\.err}
python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 32 64 128 256 --epochs 10 50 100 500 1> $Lfile 2> $Efile &
#14092
# grep -P ',[12]' /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/Sun_etal_dataset.fingerprints.hpTuningResults.01-EpochsBatchSizeAromatase.csv

#  4, 10, 64,0.8972122430801391,0.010573020601895697,2
# 14,100,256,0.89823077917099,0.0068985283970507865,1

#### now, lets tune optimizers

LfileO="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O.256.100.diverse."$target".log"
EfileO=${LfileO//\.log/\.err}

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 256 --epochs 100 --optimizers 'SGD' 'RMSprop' 'Adagrad' 'Adadelta' 'Adam' 'Adamax' 'Nadam'  1> $LfileO 2> $EfileO &
#22905

# grep -P ',[12]' /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/Sun_etal_dataset.fingerprints.hpTuningResults.02-OptimizerAromatase.csv

# 3,Adadelta,0.8976212263107299,0.005689904657439976,1
# 4,    Adam,0.8943726897239686,0.004946206947486227,2

#### now, lets tune activation functions

LfileA="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O-A.64.100.Adadelta.diverse."$target".log"
EfileA=${LfileA//\.log/\.err}
#[detached from 22945.ERtuning]

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 64 --epochs 100 --optimizers 'Adadelta' --activations 'sigmoid' 'relu'  1> $LfileA 2> $EfileA &
# 22742
### So, I decided to fix the batch_size and epochs to 128 and 250, resp.

# tune optimizer again

LfileO="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O.128.250.diverse."$target".log"
EfileO=${LfileO//\.log/\.err}

python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 128 --epochs 250 --optimizers 'SGD' 'RMSprop' 'Adagrad' 'Adadelta' 'Adam' 'Adamax' 'Nadam'  1> $LfileO 2> $EfileO &
# 3285

LfileA="/data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/HPtuning/tuning-BS-E-O-A.128.100.Adadelta.diverse."$target".log"
EfileA=${LfileA//\.log/\.err}
python ./deepFPlearn-HyperParameterTuning_single.py -i $input -t $target -p $outpath --batchSizes 128 --epochs 100 --optimizers 'Adadelta' --activations 'sigmoid' 'relu'  1> $LfileA 2> $EfileA &


#[detached from 12873.AROtuning] --------------------------------------------------------






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
