#!/bin/bash

#$ -N dfpl_train_cpu
#$ -S /bin/bash
#$ -l h_rt=168:00:00
#$ -l h_vmem=8G
#$ -pe smp 2-12

# output files
#$ -o /work/$USER/$JOB_NAME.out
#$ -e /work/$USER/$JOB_NAME.err

ml purge
ml anaconda/5/5.0.1

conda activate rdkit2019

python /home/hertelj/git-hertelj/code/2019_deepFPlearn/deepFPlearn train -i /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/input/Sun_etal_dataset.csv -o /data/bioinf/projects/data/2019_IDA-chem/deepFPlearn/modeltraining/kfoldCV_CPU/ -t 'smiles' -k 'topological' -s 2048 -a -d 256 -e 2000 -l 0.2 -K 5 -v 2
